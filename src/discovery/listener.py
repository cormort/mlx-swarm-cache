"""
listener.py — Coordinator mDNS 監聽器

在區域網路中持續監聽 `_mlx-swarm._tcp.local.` 類型的 mDNS 廣播。
當 Worker 節點上線或離線時，動態更新可用節點清單。

提供：
  - get_node_urls(): 回傳依 start_layer 排序的 Worker URL 清單
  - get_nodes_info(): 回傳所有已偵測到的節點詳細資訊

使用方式：
  listener = SwarmListener()
  listener.start()
  ...
  urls = listener.get_node_urls()  # 動態取得已排序的 Worker 清單
  ...
  listener.stop()
"""

import socket
import threading

from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

SERVICE_TYPE = "_mlx-swarm._tcp.local."


class _NodeInfo:
    """內部資料結構：保存單一 Worker 節點的資訊。"""

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.start_layer = start_layer
        self.end_layer = end_layer

    @property
    def forward_url(self) -> str:
        """產生標準的 /forward API URL。"""
        return f"http://{self.host}:{self.port}/forward"

    def to_dict(self) -> dict:
        """將節點資訊轉為字典（供 API 回傳用）。"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "forward_url": self.forward_url,
        }


class SwarmListener(ServiceListener):
    """Coordinator 的 mDNS 服務監聽器，自動偵測 Worker 節點上下線。"""

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeInfo] = {}
        self._lock = threading.Lock()
        self._zeroconf: Zeroconf | None = None
        self._browser: ServiceBrowser | None = None

    def start(self) -> None:
        """開始在區域網路中監聽 Worker 節點的 mDNS 廣播。"""
        self._zeroconf = Zeroconf()
        self._browser = ServiceBrowser(self._zeroconf, SERVICE_TYPE, self)
        print(f"👂 Coordinator 已開始監聽 mDNS 廣播 (類型: {SERVICE_TYPE})")

    def stop(self) -> None:
        """停止監聽並釋放資源。"""
        if self._zeroconf:
            self._zeroconf.close()
            self._zeroconf = None
            self._browser = None
            print("👂 Coordinator mDNS 監聽已關閉")

    # ── ServiceListener 介面實作 ─────────────────────────────────────────

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """當偵測到新的 Worker 節點上線時觸發。"""
        info = zc.get_service_info(type_, name)
        if info is None:
            return

        node_id = self._decode_property(info.properties, "node_id", name)
        start_layer = int(
            self._decode_property(info.properties, "start_layer", "0")
        )
        end_layer = int(
            self._decode_property(info.properties, "end_layer", "0")
        )

        # 解析 IP 位址 (安全寫法：支援 IPv4 與 IPv6)
        parsed_ips = info.parsed_addresses()
        if parsed_ips:
            host = parsed_ips[0]
        else:
            host = info.server.rstrip(".")

        port = info.port

        node = _NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        with self._lock:
            self._nodes[name] = node

        print(
            f"🟢 發現新節點: [{node_id}] "
            f"({host}:{port}, Layers: {start_layer}-{end_layer - 1})"
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """當 Worker 節點離線時觸發。"""
        with self._lock:
            node = self._nodes.pop(name, None)

        if node:
            print(f"🔴 節點離線: [{node.node_id}]")

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """當 Worker 節點資訊更新時觸發（重新解析）。"""
        self.add_service(zc, type_, name)

    # ── 公開 API ────────────────────────────────────────────────────────

    def get_node_urls(self) -> list[str]:
        """取得依 start_layer 排序的 Worker 節點 Forward URL 清單。

        Returns:
            依照神經網路層順序排列的 URL 清單，例如：
            ["http://192.168.1.10:8000/forward", "http://192.168.1.11:8001/forward"]
        """
        with self._lock:
            sorted_nodes = sorted(
                self._nodes.values(), key=lambda n: n.start_layer
            )
            return [n.forward_url for n in sorted_nodes]

    def get_nodes_info(self) -> list[dict]:
        """取得所有已偵測到的節點詳細資訊。

        Returns:
            依照 start_layer 排序的節點資訊字典清單。
        """
        with self._lock:
            sorted_nodes = sorted(
                self._nodes.values(), key=lambda n: n.start_layer
            )
            return [n.to_dict() for n in sorted_nodes]

    def remove_node_by_url(self, url: str) -> None:
        """將指定 URL 的殭屍節點強制從清單中剔除。

        Args:
            url: Worker 的 Forward URL（例如 http://192.168.1.10:8000/forward）
        """
        with self._lock:
            # 找到符合該 URL 的節點 key 並刪除
            target_key: str | None = None
            for key, node in self._nodes.items():
                if node.forward_url == url:
                    target_key = key
                    break

            if target_key:
                node = self._nodes.pop(target_key)
                print(f"👻 偵測到殭屍節點，已強制剔除: [{node.node_id}]")

    @property
    def node_count(self) -> int:
        """目前已偵測到的節點數量。"""
        with self._lock:
            return len(self._nodes)

    # ── 私有工具 ────────────────────────────────────────────────────────

    @staticmethod
    def _decode_property(
        properties: dict[bytes, bytes | None] | None,
        key: str,
        default: str,
    ) -> str:
        """安全地從 mDNS properties 中解碼指定的 key。

        Zeroconf 的 properties 可能是 bytes key/value，
        此方法統一處理 bytes/str 混合的情況。

        Args:
            properties: mDNS ServiceInfo 的 properties 字典。
            key: 要擷取的 key 名稱。
            default: 找不到時的預設值。

        Returns:
            解碼後的字串值。
        """
        if not properties:
            return default

        key_bytes = key.encode("utf-8")
        raw_value = properties.get(key_bytes)

        if raw_value is None:
            return default
        if isinstance(raw_value, bytes):
            return raw_value.decode("utf-8")
        return str(raw_value)
