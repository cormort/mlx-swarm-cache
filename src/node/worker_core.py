"""
worker_core.py — ExoWorkerNode

模擬 exo 工作節點：接收上游傳來的特徵矩陣（hidden states），
依序通過本機負責的神經網路層，並透過 AsyncTieredKVCache 管理記憶體。

修正紀錄 v2
-----------
- [Bug #4] 每層一條 I/O 執行緒：16 層 = 16 條執行緒同時競爭磁碟
  → ExoWorkerNode 層級建立一個共用 io_queue 與一條共用 I/O 執行緒，
    所有層的 AsyncTieredKVCache 共享此 Queue。
- [Bug #7] __init__ 建立執行緒後若拋出例外，已啟動的執行緒會 orphan
  → try/except 包住初始化，失敗時呼叫 shutdown() 清理。
"""

import mlx.core as mx
import queue
import threading
import time

from src.cache.async_tiered_cache import AsyncTieredKVCache


class ExoWorkerNode:
    def __init__(
        self,
        node_id: str,
        assigned_layers: range,
        max_ram_blocks: int = 5,
    ):
        """
        Args:
            node_id:         節點唯一識別名稱（例如 "mac_mini_m4"）。
            assigned_layers: 本節點負責的神經網路層範圍（例如 range(0, 16)）。
            max_ram_blocks:  每層 KV Cache 最多保留在 RAM 中的區塊數。
        """
        self.node_id = node_id
        self.assigned_layers = list(assigned_layers)
        self.layer_caches: dict[int, AsyncTieredKVCache] = {}

        print(f"🚀 啟動 Exo 節點 [{self.node_id}] - 負責層級: {self.assigned_layers}")

        # ── Bug #4 修正：整個節點共用一條 I/O 執行緒 ──────────────────────
        # 所有層共用同一個 Queue；一條執行緒循序消費，避免多執行緒競爭磁碟。
        self._shared_io_queue: queue.Queue = queue.Queue()
        self._shared_io_thread = threading.Thread(
            target=self._shared_io_worker_loop,
            daemon=True,
            name=f"SSD-IO-{node_id}",
        )
        self._shared_io_thread.start()
        print(f"  🛠️  [{self.node_id}] 共用 SSD I/O 執行緒已啟動")

        # ── Bug #7 修正：初始化失敗時清理已啟動的執行緒 ──────────────────
        try:
            for layer_idx in self.assigned_layers:
                self.layer_caches[layer_idx] = AsyncTieredKVCache(
                    max_ram_blocks=max_ram_blocks,
                    cache_dir=f"./{self.node_id}_cache/layer_{layer_idx}",
                    io_queue=self._shared_io_queue,   # 共用 Queue
                )
        except Exception:
            self.shutdown()
            raise

    # ── 共用 I/O 執行緒主迴圈 ────────────────────────────────────────────

    def _shared_io_worker_loop(self) -> None:
        """
        消費所有層共用的 io_queue。
        task 格式：(block_id, tensors, filepath, on_written) 或 None（結束信號）。
        """
        while True:
            task = self._shared_io_queue.get()
            if task is None:
                self._shared_io_queue.task_done()
                break
            block_id, tensors, filepath, on_written = task
            start_write = time.time()
            try:
                import mlx.core as _mx
                _mx.save_safetensors(filepath, tensors)
            except Exception as e:
                print(f"  [共用 I/O] ❌ Block {block_id} 寫入失敗: {e}")
                self._shared_io_queue.task_done()
                continue
            write_time = time.time() - start_write
            on_written(block_id)
            print(
                f"  [共用 I/O] 💾 Block {block_id} 已寫入 SSD "
                f"(耗時 {write_time:.4f} 秒)"
            )
            self._shared_io_queue.task_done()

    # ── 推理 ──────────────────────────────────────────────────────────────

    def forward_pass(self, hidden_states: mx.array, current_block_id: str) -> mx.array:
        """
        執行本節點負責的所有層的前向傳播。

        Args:
            hidden_states:    從上游節點（或 Coordinator）傳入的特徵矩陣。
            current_block_id: 本次推理的 Block 識別碼（對應 KV Cache 索引鍵）。

        Returns:
            處理完畢的特徵矩陣，準備送往下一個節點。
        """
        print(
            f"\n[{self.node_id}] 📥 收到網路傳來的特徵，"
            f"開始處理 Block: {current_block_id}"
        )
        x = hidden_states
        start_time = time.time()

        for layer_idx in self.assigned_layers:
            cache = self.layer_caches[layer_idx]

            # ── 1. Q/K/V 投影（PoC 使用隨機張量模擬） ──────────────────────
            # 實務替換為：q, k, v = self.model.layers[layer_idx].attention(x)
            shape = (1, 32, 128, 128)
            new_k = mx.random.uniform(shape=shape)
            new_v = mx.random.uniform(shape=shape)

            # ── 2. 寫入 AsyncTieredKVCache（自動 LRU 非同步卸載）──────────
            cache.put_block(current_block_id, new_k, new_v)

            # ── 3. 讀取歷史 KV 計算 Attention（可能觸發 SSD 同步讀取）──────
            past_k, past_v = cache.get_block(current_block_id)

            # ── 4. Attention + FFN（PoC 以加法代替） ─────────────────────
            # 實務替換為：x = mlx_attention_and_ffn(x, q, past_k, past_v)
            x = x + mx.random.uniform(shape=x.shape)

        mx.eval(x)
        compute_time = time.time() - start_time
        print(
            f"[{self.node_id}] 📤 處理完成，耗時 {compute_time:.4f} 秒。"
            " 準備將資料透過網路傳給下一個節點..."
        )
        return x

    def shutdown(self) -> None:
        """優雅關閉共用 I/O 執行緒（等待所有寫入任務完成後再結束）。"""
        self._shared_io_queue.put(None)
        self._shared_io_thread.join()
        print(f"[{self.node_id}] 🛑 共用 I/O 執行緒已關閉。")


# ─────────────────────────────────────────────
# 自測：模擬兩台 Mac 叢集
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mac_mini = ExoWorkerNode(
        node_id="mac_mini",
        assigned_layers=range(0, 16),
        max_ram_blocks=2,
    )
    macbook_air = ExoWorkerNode(
        node_id="macbook_air",
        assigned_layers=range(16, 32),
        max_ram_blocks=2,
    )

    initial_hidden_states = mx.random.uniform(shape=(1, 1024, 4096))

    for block_id in ["Block_1", "Block_2", "Block_3"]:
        print(f"\n========== 🌐 開始處理 {block_id} ==========")
        mid_hidden_states = mac_mini.forward_pass(initial_hidden_states, block_id)
        final_output = macbook_air.forward_pass(mid_hidden_states, block_id)
        print(f"========== 🏁 {block_id} 完成，產出最終 Token ==========")

    mac_mini.shutdown()
    macbook_air.shutdown()
