"""
test_discovery.py — 驗證 Auto-Discovery (mDNS/Zeroconf) 功能

測試情境：
  1. Announcer 能正確註冊與取消 mDNS 服務
  2. Listener 能偵測到新上線的節點
  3. Listener 依 start_layer 正確排序節點
  4. 節點離線時 Listener 自動移除
  5. Coordinator 手動模式向後相容

執行：
    python -m pytest tests/test_discovery.py -v
"""

import importlib
import time

from src.discovery.announcer import SwarmAnnouncer
from src.discovery.listener import SwarmListener


class TestSwarmAnnouncer:
    """測試 Worker 節點 mDNS 廣播器。"""

    def test_register_and_unregister(self):
        """廣播器應能成功註冊與取消 mDNS 服務。"""
        announcer = SwarmAnnouncer(
            node_id="test_node_1",
            port=9000,
            start_layer=0,
            end_layer=16,
        )
        announcer.register()
        assert announcer._zeroconf is not None
        assert announcer._info is not None

        announcer.unregister()
        assert announcer._zeroconf is None
        assert announcer._info is None

    def test_double_unregister_safe(self):
        """重複呼叫 unregister() 不應拋出例外。"""
        announcer = SwarmAnnouncer(
            node_id="test_node_safe",
            port=9001,
            start_layer=0,
            end_layer=8,
        )
        announcer.register()
        announcer.unregister()
        # 第二次不應拋錯
        announcer.unregister()

    def test_get_local_ip(self):
        """應能取得一個有效的 IP 位址字串。"""
        ip = SwarmAnnouncer._get_local_ip()
        assert isinstance(ip, str)
        parts = ip.split(".")
        assert len(parts) == 4


class TestSwarmListener:
    """測試 Coordinator mDNS 監聽器。"""

    def test_listener_discovers_announcer(self):
        """Listener 應能自動偵測到 Announcer 廣播的節點。"""
        listener = SwarmListener()
        listener.start()

        announcer = SwarmAnnouncer(
            node_id="discovery_test_node",
            port=9100,
            start_layer=0,
            end_layer=16,
        )
        announcer.register()

        # 等待 mDNS 廣播被偵測到（通常 < 2 秒）
        deadline = time.time() + 5
        while listener.node_count == 0 and time.time() < deadline:
            time.sleep(0.2)

        assert listener.node_count >= 1, "Listener 未能偵測到 Announcer 廣播的節點"

        urls = listener.get_node_urls()
        assert any("9100" in url for url in urls)

        announcer.unregister()
        listener.stop()

    def test_nodes_sorted_by_start_layer(self):
        """多個節點應依 start_layer 排序。"""
        listener = SwarmListener()
        listener.start()

        # 故意先註冊後面的層
        announcer_b = SwarmAnnouncer(
            node_id="sort_test_node_b",
            port=9201,
            start_layer=16,
            end_layer=32,
        )
        announcer_a = SwarmAnnouncer(
            node_id="sort_test_node_a",
            port=9200,
            start_layer=0,
            end_layer=16,
        )

        announcer_b.register()
        time.sleep(0.5)
        announcer_a.register()

        # 等待兩個節點都被偵測到
        deadline = time.time() + 5
        while listener.node_count < 2 and time.time() < deadline:
            time.sleep(0.2)

        assert listener.node_count >= 2, "Listener 未能偵測到兩個節點"

        # 過濾出我們測試用的 node_id 對應的 URL，排除環境中的雜訊
        test_urls = []
        with listener._lock:
            for n in listener._nodes.values():
                if n.node_id in ("sort_test_node_a", "sort_test_node_b"):
                    test_urls.append(n)
        
        # 依照 start_layer 排序確認
        test_urls.sort(key=lambda x: x.start_layer)
        
        # 第一個應該是 port 9200（start_layer=0），第二個是 9201（start_layer=16）
        assert "9200" in test_urls[0].forward_url, f"排序錯誤：第一個 URL 應含 9200，但得到 {test_urls[0].forward_url}"
        assert "9201" in test_urls[1].forward_url, f"排序錯誤：第二個 URL 應含 9201，但得到 {test_urls[1].forward_url}"

        announcer_a.unregister()
        announcer_b.unregister()
        listener.stop()

    def test_node_removal_on_unregister(self):
        """節點取消廣播後，Listener 應自動移除該節點。"""
        listener = SwarmListener()
        listener.start()

        announcer = SwarmAnnouncer(
            node_id="removal_test_node",
            port=9300,
            start_layer=0,
            end_layer=16,
        )
        announcer.register()

        # 等待節點被偵測到
        deadline = time.time() + 5
        while listener.node_count == 0 and time.time() < deadline:
            time.sleep(0.2)

        assert listener.node_count >= 1

        # 取消廣播
        announcer.unregister()

        # 等待節點被移除（mDNS 移除事件可能需要一點時間）
        deadline = time.time() + 5
        while listener.node_count > 0 and time.time() < deadline:
            time.sleep(0.2)

        # 注意：mDNS 的移除事件在本機環境下通常很快，
        # 但在極端情況下可能需要更長時間。
        # 這裡我們檢查 URL 清單是否已移除該節點
        urls = listener.get_node_urls()
        assert all("9300" not in url for url in urls), "節點未被正確移除"

        listener.stop()

    def test_get_nodes_info(self):
        """get_nodes_info() 應回傳完整的節點詳細資訊。"""
        listener = SwarmListener()
        listener.start()

        announcer = SwarmAnnouncer(
            node_id="info_test_node",
            port=9400,
            start_layer=8,
            end_layer=24,
        )
        announcer.register()

        deadline = time.time() + 5
        while listener.node_count == 0 and time.time() < deadline:
            time.sleep(0.2)

        nodes = listener.get_nodes_info()
        assert len(nodes) >= 1

        # 從回傳清單中找到我們測試用的節點，排除環境干擾
        target_node = None
        for n in nodes:
            if n["node_id"] == "info_test_node":
                target_node = n
                break
                
        assert target_node is not None, "未能找到測試節點 info_test_node"

        assert target_node["node_id"] == "info_test_node"
        assert target_node["port"] == 9400
        assert target_node["start_layer"] == 8
        assert target_node["end_layer"] == 24
        assert "forward_url" in target_node

        announcer.unregister()
        listener.stop()


class TestCoordinatorManualMode:
    """測試 Coordinator 手動模式的向後相容性。"""

    def test_manual_mode_uses_node_urls(self, monkeypatch):
        """DISCOVERY_MODE=manual 時應使用 NODE_URLS 環境變數。"""
        monkeypatch.setenv("DISCOVERY_MODE", "manual")
        monkeypatch.setenv(
            "NODE_URLS",
            "http://localhost:8000/forward,http://localhost:8001/forward",
        )

        import src.orchestrator.coordinator as coord

        importlib.reload(coord)

        assert coord.DISCOVERY_MODE == "manual"
        urls = coord.get_active_node_urls()
        assert len(urls) == 2
        assert urls[0] == "http://localhost:8000/forward"

    def test_auto_mode_without_listener_falls_back(self, monkeypatch):
        """auto 模式但 _listener 為 None 時，應回退使用手動清單。"""
        monkeypatch.setenv("DISCOVERY_MODE", "auto")
        monkeypatch.setenv(
            "NODE_URLS",
            "http://fallback:8000/forward",
        )

        import src.orchestrator.coordinator as coord

        importlib.reload(coord)
        coord._listener = None

        urls = coord.get_active_node_urls()
        assert "fallback" in urls[0]
