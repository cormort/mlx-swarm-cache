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

import logging
import os
import socket

from zeroconf import ServiceInfo, Zeroconf

logger = logging.getLogger("mlx-swarm")

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
            parsed_addresses=[local_ip],  # 取代舊版的 inet_aton 與 addresses，安全支援 IPv4/v6
            port=self.port,
            properties={
                "node_id": self.node_id,
                "start_layer": str(self.start_layer),
                "end_layer": str(self.end_layer),
            },
        )

        self._zeroconf.register_service(self._info)
        logger.info(
            "📡 [%s] mDNS 廣播已啟動 (IP: %s, Port: %d, Layers: %d-%d)",
            self.node_id, local_ip, self.port, self.start_layer, self.end_layer - 1,
        )

    def unregister(self) -> None:
        """取消發佈 mDNS 廣播並釋放相關資源。

        在 Worker 程式結束（Ctrl+C 或發生例外）時必須呼叫，
        以確保 Coordinator 能夠及時發現節點離線。加入防呆保護
        機制，防止註冊過程錯誤導致 unregister 異常崩潰。
        """
        if self._zeroconf and self._info:
            try:
                self._zeroconf.unregister_service(self._info)
            except Exception as e:
                logger.warning("⚠️ mDNS 服務解除註冊發生錯誤: %s", e)
            finally:
                self._zeroconf.close()
                self._zeroconf = None
                self._info = None
                logger.info("👋 已停止 mDNS 廣播 Node [%s]", self.node_id)

    @staticmethod
    def _get_local_ip() -> str:
        """偵測本機的區域網路 IP 位址。

        為解決 Thunderbolt Bridge (169.254.x.x) 與 Wi-Fi 網路的路由衝突，
        優先檢查作業系統環境變數 HOST_IP 是否有被手動指定。否則使用預設
        UDP Socket 策略進行偵測。

        Returns:
            本機對區網可見的有效 IP 字串。
        """
        # 1. 優先讀取環境變數，解決 Thunderbolt 等特定網卡綁定需求
        forced_ip = os.environ.get("HOST_IP")
        if forced_ip:
            return forced_ip

        # 2. 自動偵測邏輯
        try:
            # 建立一個 UDP socket 並連接到路由專用 IP
            # (此 IP 不需要真的存在，用途是讓 OS 告訴我們它會用哪張網卡發送封包)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("10.255.255.255", 1))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except Exception:
            # 如果無法偵測，則回傳本機迴路位址
            return "127.0.0.1"
