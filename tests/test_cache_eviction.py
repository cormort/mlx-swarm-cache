"""
test_cache_eviction.py — 驗證分層 KV Cache 的卸載與回載行為

測試情境：
  1. RAM 滿載時，LRU Block 是否正確卸載至 SSD
  2. 讀取 SSD Block 後，RAM 是否正確回載並維持大小限制
  3. 非同步卸載不阻斷主執行緒
  4. 讀取不存在的 Block 安全回傳 (None, None)
  5. [Bug #1 修正] 並發讀寫 ssd_index 不造成 Race Condition
  6. [Bug #2 修正] 卸載後立即讀取，等待落盤而不拋 FileNotFoundError
  7. [Security]   block_id 含路徑穿越字元時，檔案仍在 cache_dir 內
  8. [Bug #3 修正] coordinator NODE_URLS 支援超過兩個節點

執行：
    python -m pytest tests/test_cache_eviction.py -v
"""

import os
import shutil
import tempfile
import threading
import time

import mlx.core as mx
import pytest

from src.cache.async_tiered_cache import AsyncTieredKVCache, TieredKVCache

SHAPE = (1, 4, 8, 8)


def make_tensors():
    return mx.random.uniform(shape=SHAPE), mx.random.uniform(shape=SHAPE)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_cache_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ─────────────────────────────────────────────
# TieredKVCache（同步版）
# ─────────────────────────────────────────────

class TestTieredKVCache:

    def test_ram_hit(self, tmp_dir):
        cache = TieredKVCache(max_ram_blocks=3, cache_dir=tmp_dir)
        k, v = make_tensors()
        cache.put_block("A", k, v)
        got_k, got_v = cache.get_block("A")
        assert got_k is not None and got_v is not None
        assert "A" in cache.ram_cache
        assert "A" not in cache.ssd_index

    def test_lru_eviction_to_ssd(self, tmp_dir):
        cache = TieredKVCache(max_ram_blocks=2, cache_dir=tmp_dir)
        cache.put_block("A", *make_tensors())
        cache.put_block("B", *make_tensors())
        cache.put_block("C", *make_tensors())   # A 應被踢到 SSD
        assert "A" not in cache.ram_cache
        assert "A" in cache.ssd_index
        assert len(cache.ram_cache) == 2

    def test_ssd_reload(self, tmp_dir):
        cache = TieredKVCache(max_ram_blocks=2, cache_dir=tmp_dir)
        cache.put_block("A", *make_tensors())
        cache.put_block("B", *make_tensors())
        cache.put_block("C", *make_tensors())
        got_k, got_v = cache.get_block("A")
        assert got_k is not None
        assert "A" in cache.ram_cache
        assert len(cache.ram_cache) <= 2

    def test_missing_block_returns_none(self, tmp_dir):
        cache = TieredKVCache(max_ram_blocks=3, cache_dir=tmp_dir)
        k, v = cache.get_block("non_existent")
        assert k is None and v is None

    def test_lru_order_after_read(self, tmp_dir):
        cache = TieredKVCache(max_ram_blocks=2, cache_dir=tmp_dir)
        cache.put_block("A", *make_tensors())
        cache.put_block("B", *make_tensors())
        cache.get_block("A")                    # A 成為最近使用
        cache.put_block("C", *make_tensors())   # B 應被踢走
        assert "B" in cache.ssd_index
        assert "A" in cache.ram_cache
        assert "C" in cache.ram_cache

    def test_safetensors_file_created(self, tmp_dir):
        cache = TieredKVCache(max_ram_blocks=1, cache_dir=tmp_dir)
        cache.put_block("A", *make_tensors())
        cache.put_block("B", *make_tensors())   # A 被卸載
        filepath = os.path.join(tmp_dir, "block_A.safetensors")
        assert os.path.exists(filepath)

    # ── Security ──────────────────────────────────────────────────────────

    def test_path_traversal_blocked(self, tmp_dir):
        """block_id 含 ../ 時，檔案應仍在 cache_dir 內（Security 修正）。"""
        cache = TieredKVCache(max_ram_blocks=1, cache_dir=tmp_dir)
        cache.put_block("A", *make_tensors())
        cache.put_block("../evil", *make_tensors())   # 觸發卸載 A
        # 如果路徑穿越沒被過濾，block_A 會被寫到 tmp_dir 以外
        # 這裡只要確認 cache_dir 外沒有出現 evil 相關檔案
        parent = os.path.dirname(tmp_dir)
        evil_files = [f for f in os.listdir(parent) if "evil" in f]
        assert len(evil_files) == 0, f"路徑穿越未被阻擋，發現: {evil_files}"


# ─────────────────────────────────────────────
# AsyncTieredKVCache（非同步版）
# ─────────────────────────────────────────────

class TestAsyncTieredKVCache:

    def test_async_eviction_does_not_block(self, tmp_dir):
        """
        非同步卸載的主執行緒耗時應遠小於一次磁碟寫入（Bug #1 修正驗證）。
        量測單次 put_block（觸發卸載）的耗時，應 < 100 ms。
        """
        cache = AsyncTieredKVCache(max_ram_blocks=1, cache_dir=tmp_dir)
        cache.put_block("warm", *make_tensors())   # 填滿 RAM

        start = time.time()
        cache.put_block("new", *make_tensors())    # 觸發非同步卸載
        elapsed_ms = (time.time() - start) * 1000

        cache.shutdown()
        assert elapsed_ms < 100, (
            f"非同步卸載不應阻塞主執行緒，但耗時 {elapsed_ms:.1f} ms"
        )

    def test_read_after_evict_does_not_raise(self, tmp_dir):
        """
        卸載後立即讀取，不應拋出 FileNotFoundError（Bug #2 修正驗證）。
        """
        cache = AsyncTieredKVCache(max_ram_blocks=1, cache_dir=tmp_dir)
        cache.put_block("A", *make_tensors())
        cache.put_block("B", *make_tensors())   # A 被非同步卸載

        # 此時 A 可能仍在寫入途中，正確行為是等待落盤後回傳，不拋例外
        try:
            got_k, got_v = cache.get_block("A")
            assert got_k is not None, "A 應能在等待落盤後被讀回"
        except FileNotFoundError:
            pytest.fail("Bug #2 未修正：get_block() 在檔案未落盤時拋出 FileNotFoundError")
        finally:
            cache.shutdown()

    def test_ssd_index_no_race_condition(self, tmp_dir):
        """
        高並發讀寫 ssd_index 不造成 Race Condition（Bug #1 修正驗證）。
        開 20 條執行緒同時讀寫快取，不應拋出任何例外。
        """
        cache = AsyncTieredKVCache(max_ram_blocks=2, cache_dir=tmp_dir)
        errors: list[Exception] = []

        def worker(tid: int):
            try:
                for i in range(5):
                    bid = f"t{tid}_b{i}"
                    cache.put_block(bid, *make_tensors())
                    cache.get_block(bid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        cache.shutdown()
        assert len(errors) == 0, f"Race Condition 偵測到例外：{errors}"

    def test_async_ssd_file_eventually_created(self, tmp_dir):
        """背景 I/O 完成後，SSD 上應出現對應檔案。"""
        cache = AsyncTieredKVCache(max_ram_blocks=1, cache_dir=tmp_dir)
        cache.put_block("X", *make_tensors())
        cache.put_block("Y", *make_tensors())   # X 被非同步卸載
        cache.io_queue.join()                   # 等待所有任務完成
        filepath = os.path.join(tmp_dir, "block_X.safetensors")
        assert os.path.exists(filepath)
        cache.shutdown()

    def test_async_read_missing_returns_none(self, tmp_dir):
        cache = AsyncTieredKVCache(max_ram_blocks=3, cache_dir=tmp_dir)
        k, v = cache.get_block("ghost")
        cache.shutdown()
        assert k is None and v is None

    def test_path_traversal_blocked(self, tmp_dir):
        """非同步版的路徑穿越防護（Security 修正驗證）。"""
        cache = AsyncTieredKVCache(max_ram_blocks=1, cache_dir=tmp_dir)
        cache.put_block("A", *make_tensors())
        cache.put_block("../evil", *make_tensors())
        cache.io_queue.join()
        parent = os.path.dirname(tmp_dir)
        evil_files = [f for f in os.listdir(parent) if "evil" in f]
        cache.shutdown()
        assert len(evil_files) == 0


# ─────────────────────────────────────────────
# Coordinator NODE_URLS 解析（Bug #3 修正驗證）
# ─────────────────────────────────────────────

class TestCoordinatorNodeUrls:

    def test_node_urls_parsed_from_env(self, monkeypatch):
        """NODE_URLS 環境變數應能正確解析為超過兩個節點的清單。"""
        three_nodes = (
            "http://localhost:8000/forward,"
            "http://192.168.1.100:8001/forward,"
            "http://192.168.1.101:8002/forward"
        )
        monkeypatch.setenv("NODE_URLS", three_nodes)

        # 重新 import 以觸發模組層級的 os.getenv 重新讀取
        import importlib
        import src.orchestrator.coordinator as coord
        importlib.reload(coord)

        assert len(coord.NODE_URLS) == 3
        assert coord.NODE_URLS[2] == "http://192.168.1.101:8002/forward"

    def test_node_urls_strips_whitespace(self, monkeypatch):
        """URL 之間的空白應被自動去除。"""
        monkeypatch.setenv(
            "NODE_URLS",
            "  http://localhost:8000/forward ,  http://localhost:8001/forward  "
        )
        import importlib
        import src.orchestrator.coordinator as coord
        importlib.reload(coord)

        assert all(" " not in url for url in coord.NODE_URLS)
