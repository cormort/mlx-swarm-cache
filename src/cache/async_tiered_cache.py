"""
async_tiered_cache.py — 分層 KV Cache（含同步與非同步版本）

步驟一：TieredKVCache      — 同步 LRU 卸載，用於理解基本機制
步驟三：AsyncTieredKVCache — 背景執行緒非同步卸載，解放 MLX 主計算執行緒

修正紀錄 v2
-----------
- [Bug #1] Race Condition：ssd_index 同時被主執行緒讀、背景執行緒寫
  → 加入 threading.Lock()；所有對 ram_cache / ssd_index / pending_ssd
    的存取都在鎖內完成。
- [Bug #2] 卸載後立即讀取，檔案尚未落盤就呼叫 mx.load() → FileNotFoundError
  → 新增 pending_ssd set：block 丟進 Queue 時立刻標記；
    get_block() 遇到 pending 的 block 時輪詢等待落盤後再載入。
- [Security] block_id 直接嵌入檔名，可能路徑穿越（../）
  → _safe_block_id() 過濾非法字元。
- [Design] AsyncTieredKVCache 接受可選的外部共用 io_queue
  → ExoWorkerNode 可讓所有層共用一條 I/O 執行緒，避免 32 條執行緒
    同時競爭磁碟頻寬。
"""

import mlx.core as mx
import os
import time
import threading
import queue
from collections import OrderedDict


# ─────────────────────────────────────────────
# 步驟一：同步版本（概念驗證用）
# ─────────────────────────────────────────────

class TieredKVCache:
    def __init__(self, max_ram_blocks: int = 3, cache_dir: str = "./omlx_ssd_cache"):
        self.max_ram_blocks = max_ram_blocks
        self.cache_dir = cache_dir
        self.ram_cache: OrderedDict = OrderedDict()
        self.ssd_index: set = set()
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"📁 已建立 SSD 快取資料夾: {self.cache_dir}")

    @staticmethod
    def _safe_block_id(block_id: str) -> str:
        return block_id.replace("/", "_").replace("\\", "_").replace("..", "__")

    def _get_filepath(self, block_id: str) -> str:
        return os.path.join(
            self.cache_dir, f"block_{self._safe_block_id(block_id)}.safetensors"
        )

    def put_block(self, block_id: str, k_tensor: mx.array, v_tensor: mx.array) -> None:
        print(f"\n[寫入] 準備寫入 Block {block_id} 到 RAM...")
        if block_id in self.ram_cache:
            del self.ram_cache[block_id]
        self.ram_cache[block_id] = {"k": k_tensor, "v": v_tensor}
        if len(self.ram_cache) > self.max_ram_blocks:
            self._evict_lru_to_ssd()

    def get_block(self, block_id: str):
        if block_id in self.ram_cache:
            print(f"[讀取] Block {block_id} 命中 RAM (Hot Tier) ⚡️")
            self.ram_cache.move_to_end(block_id)
            return self.ram_cache[block_id]["k"], self.ram_cache[block_id]["v"]
        if block_id in self.ssd_index:
            print(f"[讀取] Block {block_id} 命中 SSD (Cold Tier)，準備載入 RAM 💾")
            return self._load_from_ssd(block_id)
        print(f"[讀取] 找不到 Block {block_id} ❌")
        return None, None

    def _evict_lru_to_ssd(self) -> None:
        lru_block_id, tensors = self.ram_cache.popitem(last=False)
        filepath = self._get_filepath(lru_block_id)
        mx.save_safetensors(filepath, tensors)
        self.ssd_index.add(lru_block_id)
        print(f"  ⚠️  RAM 已滿！已將 Block {lru_block_id} 卸載至 SSD: {filepath}")

    def _load_from_ssd(self, block_id: str):
        filepath = self._get_filepath(block_id)
        tensors = mx.load(filepath)
        k_tensor, v_tensor = tensors["k"], tensors["v"]
        self.put_block(block_id, k_tensor, v_tensor)
        return k_tensor, v_tensor


# ─────────────────────────────────────────────
# io_queue task 格式（所有執行緒都遵守此 4-tuple）：
#   (block_id: str,
#    tensors: dict,
#    filepath: str,
#    on_written: Callable[[str], None])
#
# on_written 是寫入完成後的回呼，由 cache 自行提供，
# I/O 執行緒只負責寫檔後呼叫它，不直接持有 cache 狀態。
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 步驟三：非同步版本（生產用）
# ─────────────────────────────────────────────

class AsyncTieredKVCache:
    """
    升級版分層 KV Cache。

    卸載（RAM → SSD）為非同步：主執行緒派發任務到 Queue 後立即繼續。
    讀取（SSD → RAM）為同步：若 block 仍在 pending_ssd，會等待落盤完成。

    執行緒安全
    ----------
    _lock 保護 ram_cache / ssd_index / pending_ssd 的所有讀寫。
    pending_ssd：block 入隊時立刻加入，落盤後移除；
                 讓 get_block() 知道「這個 block 正在寫入途中，等一下」。
    """

    _SSD_WAIT_TIMEOUT: float = 10.0
    _SSD_WAIT_POLL:    float = 0.005

    def __init__(
        self,
        max_ram_blocks: int = 3,
        cache_dir: str = "./omlx_ssd_cache_async",
        io_queue: "queue.Queue | None" = None,
    ):
        self.max_ram_blocks = max_ram_blocks
        self.cache_dir = cache_dir
        self.ram_cache: OrderedDict = OrderedDict()
        self.ssd_index: set = set()
        self.pending_ssd: set = set()
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

        self._owns_thread = io_queue is None
        self.io_queue: queue.Queue = io_queue if io_queue is not None else queue.Queue()

        if self._owns_thread:
            self.io_thread = threading.Thread(
                target=self._io_worker_loop, daemon=True, name="SSD-IO-Worker"
            )
            self.io_thread.start()
            print("🛠️  背景 SSD 寫入執行緒已啟動...")

    # ── 內部工具 ──────────────────────────────

    @staticmethod
    def _safe_block_id(block_id: str) -> str:
        return block_id.replace("/", "_").replace("\\", "_").replace("..", "__")

    def _get_filepath(self, block_id: str) -> str:
        return os.path.join(
            self.cache_dir, f"block_{self._safe_block_id(block_id)}.safetensors"
        )

    # ── 背景執行緒主迴圈 ──────────────────────

    def _io_worker_loop(self) -> None:
        """自有執行緒時使用。消費 Queue 中的寫入任務並呼叫 on_written 回呼。"""
        while True:
            task = self.io_queue.get()
            if task is None:          # Poison Pill
                self.io_queue.task_done()
                break
            block_id, tensors, filepath, on_written = task
            start_write = time.time()
            try:
                mx.save_safetensors(filepath, tensors)
            except Exception as e:
                print(f"  [背景 I/O] ❌ Block {block_id} 寫入失敗: {e}")
                self.io_queue.task_done()
                continue
            write_time = time.time() - start_write
            on_written(block_id)
            print(
                f"  [背景 I/O] 💾 成功將 Block {block_id} 寫入 SSD "
                f"(耗時 {write_time:.4f} 秒)"
            )
            self.io_queue.task_done()

    # ── 公開 API ──────────────────────────────

    def put_block(self, block_id: str, k_tensor: mx.array, v_tensor: mx.array) -> None:
        with self._lock:
            if block_id in self.ram_cache:
                del self.ram_cache[block_id]
            self.ram_cache[block_id] = {"k": k_tensor, "v": v_tensor}
            if len(self.ram_cache) > self.max_ram_blocks:
                self._async_evict_lru_to_ssd_locked()

    def get_block(self, block_id: str):
        """
        獲取 KV 區塊。
        - RAM 命中         → 更新 LRU，立即返回
        - pending_ssd 中   → 等待落盤後再載入（不回傳 None）
        - ssd_index 中     → 從磁碟同步載回
        - 找不到           → 返回 (None, None)
        """
        with self._lock:
            if block_id in self.ram_cache:
                self.ram_cache.move_to_end(block_id)
                return self.ram_cache[block_id]["k"], self.ram_cache[block_id]["v"]
            # 在鎖內快照，在鎖外執行耗時 I/O
            in_pending = block_id in self.pending_ssd
            in_ssd     = block_id in self.ssd_index

        if in_pending or in_ssd:
            return self._sync_load_from_ssd(block_id)

        return None, None

    def shutdown(self) -> None:
        """優雅關閉背景執行緒（僅在 _owns_thread 時有效）。"""
        if self._owns_thread:
            self.io_queue.put(None)
            self.io_thread.join()
            print("🛑 背景 SSD 執行緒已安全關閉。")

    # ── 私有方法 ──────────────────────────────

    def _on_written(self, block_id: str) -> None:
        """I/O 執行緒完成寫入後的回呼：在鎖內更新 ssd_index 與 pending_ssd。"""
        with self._lock:
            self.ssd_index.add(block_id)
            self.pending_ssd.discard(block_id)

    def _async_evict_lru_to_ssd_locked(self) -> None:
        """
        非同步卸載（必須在 _lock 持有狀態下呼叫）。

        1. 從 RAM 移除 LRU block
        2. mx.eval() 確保 tensor 具體化，背景執行緒不會觸發 lazy evaluation
        3. 加入 pending_ssd（在入隊前），讓 get_block() 知道要等
        4. 派發任務到 Queue 後立即返回
        """
        lru_block_id, tensors = self.ram_cache.popitem(last=False)
        filepath = self._get_filepath(lru_block_id)
        mx.eval(tensors["k"], tensors["v"])          # ⚠️ 必須在派發前求值
        self.pending_ssd.add(lru_block_id)           # 先標記，再入隊
        self.io_queue.put((lru_block_id, tensors, filepath, self._on_written))
        print(
            f"  ⚡️ [主執行緒] 已將 Block {lru_block_id} 的卸載任務"
            " 派發給背景 Queue，繼續全速運算！"
        )

    def _sync_load_from_ssd(self, block_id: str):
        """
        同步從 SSD 讀回 RAM。

        若 block 仍在 pending_ssd（背景尚未落盤），輪詢等待檔案出現，
        最多等待 _SSD_WAIT_TIMEOUT 秒（Bug #2 修正）。
        """
        filepath = self._get_filepath(block_id)
        deadline = time.time() + self._SSD_WAIT_TIMEOUT

        while not os.path.exists(filepath):
            if time.time() > deadline:
                raise TimeoutError(
                    f"Block {block_id} 的 SSD 寫入等待逾時"
                    f"（>{self._SSD_WAIT_TIMEOUT}s），"
                    "請檢查磁碟空間或 I/O 執行緒是否正常運作。"
                )
            time.sleep(self._SSD_WAIT_POLL)

        print(f"  ⏳ [主執行緒] Block {block_id} 從 SSD 載入 RAM...")
        tensors = mx.load(filepath)
        k_tensor, v_tensor = tensors["k"], tensors["v"]
        self.put_block(block_id, k_tensor, v_tensor)
        return k_tensor, v_tensor


# ─────────────────────────────────────────────
# 快速自測（步驟一概念驗證）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cache = TieredKVCache(max_ram_blocks=2)
    shape = (1, 32, 128, 128)

    print("--- 階段 1: 連續寫入 3 個 Block ---")
    cache.put_block("A", mx.random.uniform(shape=shape), mx.random.uniform(shape=shape))
    cache.put_block("B", mx.random.uniform(shape=shape), mx.random.uniform(shape=shape))
    cache.put_block("C", mx.random.uniform(shape=shape), mx.random.uniform(shape=shape))

    print("\n--- 階段 2: 讀取測試 ---")
    cache.get_block("B")
    cache.get_block("A")
