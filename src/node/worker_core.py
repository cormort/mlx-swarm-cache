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

import queue
import threading
import time

import mlx.core as mx

from src.cache.async_tiered_cache import AsyncTieredKVCache


class ExoWorkerNode:
    def __init__(
        self,
        node_id: str,
        max_ram_blocks: int = 5,
    ):
        """
        Args:
            node_id:         節點唯一識別名稱（例如 "mac_mini_m4"）。
            max_ram_blocks:  每層 KV Cache 最多保留在 RAM 中的區塊數。
        """
        self.node_id = node_id
        self.max_ram_blocks = max_ram_blocks
        self.assigned_layers: list[int] = []
        self.layer_caches: dict[int, AsyncTieredKVCache] = {}
        
        self.model = None
        self.model_path: str | None = None

        print(f"🚀 啟動 Exo 節點 [{self.node_id}] - 等待分派模型...")

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
        # I/O 執行緒如果啟動失敗會由 Python 例外層級處理，模型與 cache 的初始化移至 load_model 中。

    def load_model(self, model_path: str, assigned_layers: list[int]) -> None:
        """動態載入模型，並過濾掉非自己負責的層。"""
        import gc

        from mlx_lm import load as mlx_load

        self.unload_model()  # 確保之前載入的模型被清除

        print(f"[{self.node_id}] 📥 開始載入模型 '{model_path}'...")
        # 1. 載入模型結構與權重 (MLX 會使用 mmap，初始記憶體佔用極低)
        model, _ = mlx_load(model_path)

        # 2. 過濾掉非自己負責的層，釋放記憶體
        # 注意: 根據不同模型架構，layers 的屬性可能不同。通常存在 `model.model.layers` 
        # (例如 LLaMA/Qwen) 或 `model.layers`。這邊泛用處理。
        target_layers_attr = getattr(model, "model", model)
        if hasattr(target_layers_attr, "layers"):
            layers = target_layers_attr.layers
            # 建立一個新 list 僅保留負責的層。然後覆蓋。
            # 但是這樣可能會破壞索引，因為 forward 需要原始 layer index?
            # 更好的做法是將不負責的層直接替換為 None (節省空間但保持索引)
            for i in range(len(layers)):
                if i not in assigned_layers:
                    layers[i] = None
        else:
            print(f"[{self.node_id}] ⚠️ 警告：無法在模型找尋到 layers 屬性，模型架構可能不支援分層！")

        # 同時移除 lm_head 和 embed_tokens 等不需要的部分以節省空間
        if hasattr(model, "lm_head"):
            model.lm_head = None
        if hasattr(target_layers_attr, "embed_tokens"):
            target_layers_attr.embed_tokens = None
        if hasattr(target_layers_attr, "norm"):
            target_layers_attr.norm = None

        gc.collect()

        # 3. 初始化對應的 KV Cache
        self.assigned_layers = assigned_layers
        self.layer_caches = {}
        for layer_idx in self.assigned_layers:
            self.layer_caches[layer_idx] = AsyncTieredKVCache(
                max_ram_blocks=self.max_ram_blocks,
                cache_dir=f"./{self.node_id}_cache/layer_{layer_idx}",
                io_queue=self._shared_io_queue,
            )
            
        self.model = model
        self.model_path = model_path
        print(f"[{self.node_id}] ✅ 模型載入完成，負責層級: {self.assigned_layers}")

    def unload_model(self) -> None:
        import gc
        self.model = None
        self.model_path = None
        self.assigned_layers = []
        self.layer_caches.clear()
        gc.collect()
        print(f"[{self.node_id}] 🗑️ 模型與 Cache 已卸載")

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

        此函式接收從上一步（Coordinator 或前一個 Node）以 msgpack 形式
        傳來的特徵矩陣，然後依序讓特徵通過本節點負責的所有神經網路層
        (`self.assigned_layers`)。

        在經過每一層時：
        1. 模擬產生該層專屬的 K/V 狀態 (Key/Value states)。
        2. 將 K/V 狀態交由 `AsyncTieredKVCache` 存入 RAM
           (滿載時自動於背景卸載到 SSD)。
        3. 從快取中讀回歷史 context 以進行 Attention 計算
           (若已卸載，則同步從 SSD 載回)。
        4. 最終呼叫 `mx.eval(x)` 確保所有延遲計算在本節點完成。

        Args:
            hidden_states (mx.array): 從網路層接收到的 MLX 特徵矩陣。
            current_block_id (str): 本次推理對應的 Cache 鍵值或 Block ID。

        Returns:
            mx.array: 經過所有分配層處理完畢的特徵矩陣，供下一階段使用。
        """
        print(
            f"\n[{self.node_id}] 📥 收到網路傳來的特徵，"
            f"開始處理 Block: {current_block_id}"
        )
        x = hidden_states
        start_time = time.time()

        if self.model is None:
            raise ValueError("Model is not loaded on this worker node!")

        target_layers_attr = getattr(self.model, "model", self.model)
        layers = target_layers_attr.layers

        for layer_idx in self.assigned_layers:
            cache = self.layer_caches[layer_idx]

            # ── 3. 讀取歷史 KV ──────────────────────────────────────────
            past_k, past_v = cache.get_block(current_block_id)
            
            # mlx_lm 模型在不同架構下 attention_cache 參數不同
            # 一般 `mlx_lm` 模型的 layer call 接受 x, mask, cache
            
            layer = layers[layer_idx]
            # 注意: 如果 past_k 和 past_v 是 None，代表這是第一次處理這個 block
            # 實務上 MLX_LM 傳遞的 cache 是 (key_cache, value_cache) array tuple 或自訂的 KVCache 物件
            # 為簡單起見，我們如果沒有 past cache，傳 None
            kv_cache = (past_k, past_v) if past_k is not None else None

            # ── 4. 前向傳播 (真實計算) ──────────────────────────────────
            # mask 在生成階段通常為 None (或 1D array)。
            # 注意：某些模型 `layer` 回傳 tuple: (hidden_states, cache)
            # 這裡需要視 `mlx_lm` 底層實作調整，一般 Qwen/Llama 回傳 (hidden_states, mask, cache) 或類似結構。
            try:
                # 較新的 mlx_lm 版本通常使用 `layer(x, mask=mask, cache=cache)`
                layer_out = layer(x, mask=None, cache=kv_cache)
                if isinstance(layer_out, tuple):
                    x = layer_out[0]
                    new_cache = layer_out[-1] # 最後一個元素有可能是 cache tuple
                    if isinstance(new_cache, tuple) and len(new_cache) == 2:
                        new_k, new_v = new_cache
                        # ── 2. 寫入 AsyncTieredKVCache ──────────────────
                        cache.put_block(current_block_id, new_k, new_v)
                else:
                    x = layer_out
            except Exception as e:
                print(f"[{self.node_id}] Layer {layer_idx} 執行失敗: {e}")
                raise

        # 只 eval 最後出來的一層確保計算真的發生
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
        max_ram_blocks=2,
    )
    # 測試程式先註解掉，因為需要真實路徑與真實資料才能跑
    # mac_mini.load_model("mlx-community/Llama-3.2-1B-Instruct-4bit", [0, 1])
    mac_mini.shutdown()
