"""
announcer.py — Worker 節點 mDNS 廣播器

當 Worker (api_server) 啟動時，透過 Zeroconf 在區域網路中註冊
一個 `_mlx-swarm._tcp.local.` 類型的服務，讓 Coordinator 能自動發現。

廣播內容包含：
  - node_id:     節點唯一識別名稱
  - start_layer: 本節點負責的起始層
  - end_layer:   本節點負責的結束層
  - port:        本節點監聽的 HTTP Port

生命週期：
  announcer = SwarmAnnouncer(...)
  announcer.register()    # 開始廣播
  ...
  announcer.unregister()  # 優雅關閉：移除廣播
"""

import socket

from zeroconf import ServiceInfo, Zeroconf

SERVICE_TYPE = "_mlx-swarm._tcp.local."


class SwarmAnnouncer:
    """Worker 節點的 mDNS 服務廣播器。"""

    def __init__(
        self,
        node_id: str,
        port: int,
        start_layer: int,
        end_layer: int,
    ) -> None:
        """初始化廣播器。

        Args:
            node_id: 節點唯一識別名稱（例如 "mac_mini_m4"）。
            port: Worker HTTP 服務監聽的 Port。
            start_layer: 本節點負責的起始神經網路層索引。
            end_layer: 本節點負責的結束神經網路層索引（不含）。
        """
        self.node_id = node_id
        self.port = port
        self.start_layer = start_layer
        self.end_layer = end_layer

        self._zeroconf: Zeroconf | None = None
        self._info: ServiceInfo | None = None

    def register(self) -> None:
        """在區域網路中註冊 mDNS 服務，開始廣播。"""
        self._zeroconf = Zeroconf()

        # 取得本機 IP（透過建立一個 UDP socket 嘗試連線來偵測）
        local_ip = self._get_local_ip()

        # 組裝 ServiceInfo
        # name 格式：{node_id}._mlx-swarm._tcp.local.
        self._info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=f"{self.node_id}.{SERVICE_TYPE}",
            server=f"{self.node_id}.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties={
                "node_id": self.node_id,
                "start_layer": str(self.start_layer),
                "end_layer": str(self.end_layer),
            },
        )

        self._zeroconf.register_service(self._info)
        print(
            f"📡 [{self.node_id}] mDNS 廣播已啟動 "
            f"(IP: {local_ip}, Port: {self.port}, "
            f"Layers: {self.start_layer}-{self.end_layer - 1})"
        )

    def unregister(self) -> None:
        """從區域網路中移除 mDNS 服務，停止廣播。"""
        if self._zeroconf and self._info:
            self._zeroconf.unregister_service(self._info)
            self._zeroconf.close()
            print(f"📡 [{self.node_id}] mDNS 廣播已關閉")
            self._zeroconf = None
            self._info = None

    @staticmethod
    def _get_local_ip() -> str:
        """偵測本機的區域網路 IP 位址。

        透過建立一個 UDP socket 並嘗試連線到外部位址（不實際傳送資料）
        來取得作業系統優先選用的網卡 IP。在 Thunderbolt Bridge 連接的環境下，
        macOS 會自動優先回傳 Thunderbolt 網卡的 IP。

        Returns:
            本機 IP 位址字串（例如 "192.168.1.10" 或 "169.254.100.2"）。
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 連線到一個不會實際傳送資料的外部位址
            sock.connect(("10.255.255.255", 1))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except Exception:
            return "127.0.0.1"
