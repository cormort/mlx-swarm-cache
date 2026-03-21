# MLX Swarm Cache 🐝

An experimental distributed LLM inference engine for Apple Silicon, combining the cluster scaling of `exo` with the tiered SSD KV caching of `omlx`.

## 核心概念

這個專案旨在解決多台 Mac 協同運算大型語言模型時，單機統一記憶體（Unified Memory）不足的痛點。
透過結合非同步的 SSD 讀寫，當叢集中的任何一個節點 RAM 滿載時，會自動將冷區塊（Cold Blocks）卸載至本機 SSD，確保分散式推理不中斷。

## 架構設計

```
使用者 Prompt
     │
     ▼
┌─────────────────────┐
│   Coordinator       │  coordinator.py
│  (指揮官節點)        │  - Tokenize & Embed
│                     │  - 排程各節點接力
└────────┬────────────┘
         │  HTTP /forward
    ┌────▼────┐       ┌─────────────┐
    │  Node 1  │──────▶│   Node 2    │
    │ Layer 0-15│      │ Layer 16-31 │
    │ Mac mini │       │ MacBook Air │
    └────┬─────┘       └──────┬──────┘
         │                    │
   ┌─────▼──────┐       ┌─────▼──────┐
   │Async Tiered│       │Async Tiered│
   │  KV Cache  │       │  KV Cache  │
   │ RAM ──► SSD│       │ RAM ──► SSD│
   └────────────┘       └────────────┘
```

1. **Coordinator**: 接收使用者輸入，轉換特徵矩陣並排程。具備 **Auto-Discovery** 能力，能透過 mDNS 自動偵測區網中的 Worker。
2. **Worker Nodes**: 每個節點是一個獨立的 FastAPI 服務，負責特定神經網路層，啟動時會自動在區網廣播自己的存在。
3. **Async Tiered Cache**: 在背景將最近最少使用（LRU）的 Context 轉為 `safetensors` 存入磁碟，釋放 RAM 給當前的運算。

## 專案結構

```
mlx-swarm-cache/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── cache/
│   │   └── async_tiered_cache.py   # 背景異步 SSD 快取機制
│   ├── discovery/
│   │   ├── announcer.py            # Worker mDNS 廣播器
│   │   └── listener.py             # Coordinator mDNS 監聽器
│   ├── node/
│   │   ├── worker_core.py          # MLX 推理與快取管理
│   │   └── api_server.py           # FastAPI 網路服務端點
│   └── orchestrator/
│       └── coordinator.py          # 指揮官：發送 Prompt 與網路排程
└── tests/
    ├── test_cache_eviction.py      # 驗證 RAM 滿載時卸載行為
    └── test_discovery.py           # 驗證 Auto-Discovery 行為
```

## 部署與執行說明 (Deployment Guide)

### 1. 準備環境 (所有節點)

首先在所有預定運行的 Mac 上複製專案並安裝依賴模組：

```bash
git clone https://github.com/cormort/mlx-swarm-cache.git
cd mlx-swarm-cache
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 啟動 Worker 節點 (Worker Nodes)

每台 Mac (或同機器的不同 Process) 負責處理不同範圍的神經網路層 (`START_LAYER` 到 `END_LAYER`)。

#### 在 Node 1 (例如 Mac mini M4, 負責前 16 層)
```bash
export NODE_ID="mac_mini_m4"
export START_LAYER=0
export END_LAYER=16
export PORT=8000
# 監聽所有網卡 0.0.0.0 (便於跨機器連線)
python -m src.node.api_server
```

#### 在 Node 2 (例如 MacBook Air, 負責後 16 層)
```bash
export NODE_ID="macbook_air"
export START_LAYER=16
export END_LAYER=32
export PORT=8001
python -m src.node.api_server
```

> **注意**：若是跨機器部署（Multi-Node），請確認各機器的防火牆 (Firewall) 允許對應 `PORT` 的 TCP 連線傳輸。

### 3. 啟動指揮官 (Coordinator API Gateway)

Coordinator 現已升級為一個 **OpenAI 相容的 FastAPI 伺服器**，並具備 **Auto-Discovery (區域網路自動尋找)** 功能。
它負責接收外部 `/v1/chat/completions` 請求，轉譯成特徵矩陣後統籌內部 Worker 網路的推論接力。

#### 模式一：自動尋找節點（推薦）
預設情況下（`DISCOVERY_MODE=auto`），Coordinator 啟動後會利用 mDNS (Zeroconf) 自動尋找區域網路內所有活著的 Worker，**完全不需要手動設定 IP**。您甚至可以隨時加入新的 Mac 來擴充算力！

```bash
# [選填] 設定外部呼叫的 API Key
export API_KEY="sk-my-secret-key-123"

# 預設監聽 0.0.0.0:8080，自動偵測 Worker
export COORDINATOR_PORT=8080 
python -m src.orchestrator.coordinator
```

查看目前已自動連結的節點：
```bash
curl http://localhost:8080/v1/nodes
```

#### 模式二：手動指定節點 (向後相容)
```bash
export DISCOVERY_MODE="manual"
export NODE_URLS="http://192.168.1.10:8000/forward,http://192.168.1.11:8001/forward"
python -m src.orchestrator.coordinator
```

### 4. 外部呼叫推論 (OpenAI Format)

當 Coordinator 啟動後，任何外部應用程式（包含支援 OpenAI Base URL 格式的套件或網頁 UI）即可像呼叫 ChatGPT API 一樣打向 Coordinator。

**Curl 測試範例：**
```bash
curl -X POST "http://localhost:8080/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-my-secret-key-123" \
     -d '{
       "model": "mlx-swarm-cache",
       "messages": [{"role": "user", "content": "請幫我寫一個結合 omlx 與 exo 的架構..."}],
       "max_tokens": 5
     }'
```

## 測試項目與結果 (Test Cases and Results)

專案包含完整的自動化測試套件，確保快取系統面對滿載、並發等極端情況都能保持功能正常與穩定，並驗證 Auto-Discovery 的可靠性。可透過下方指令執行測試：

```bash
python -m pytest tests/ -v
```

### 1. 同步快取測試 (`TieredKVCache`)
- ✅ `test_ram_hit`: 確保 RAM 快取命中時回傳正確，不觸發 SSD 讀取。
- ✅ `test_lru_eviction_to_ssd`: 測試 RAM 滿載時正確挑選並將 LRU (最近最少使用) 區塊卸載至 SSD。
- ✅ `test_ssd_reload`: 驗證已卸載的 SSD 區塊被讀取時，能自動重新載回 RAM。
- ✅ `test_missing_block_returns_none`: 請求不存在的區塊時，安全回傳 `(None, None)`。
- ✅ `test_lru_order_after_read`: 讀取操作正確刷新記憶體中區塊的 LRU 優先順序。
- ✅ `test_safetensors_file_created`: 被卸載的區塊確實以 `safetensors` 格式寫入 SSD，確保零拷貝高效載入。
- ✅ `test_path_traversal_blocked`: 安全防護：無法透過惡意命名的 `block_id` (如包含 `../`) 寫入快取專屬目錄以外的路徑。

### 2. 異步快取測試 (`AsyncTieredKVCache`)
- ✅ `test_async_eviction_does_not_block`: 確保背景 I/O 去卸載張量時不會阻塞主推理執行緒 (派發耗時 < 100ms)。
- ✅ `test_read_after_evict_does_not_raise`: 確保卸載後立刻對同一區塊進行極端頻繁讀取操作時，會安全等待檔案落盤而不會拋出 `FileNotFoundError`。
- ✅ `test_async_ssd_file_eventually_created`: 背景程序在非同步卸載觸發後，最終能在 SSD 上產生對應的 `safetensors` 檔案。
- ✅ `test_async_read_missing_returns_none`: 非同步快取版本找不到區塊時，亦能正確回傳空值。
- ✅ `test_path_traversal_blocked`: 確認異步版同樣具備防禦路徑穿越漏洞的安全控制。
- ⏭️ `test_ssd_index_no_race_condition`: *(Skipped)* 在 20 Thread 極限並發打穿卸載頻寬的壓力測試下，已知 MLX C++ 底層引擎的 `eval/load` 目前無法承受跨執行緒的交錯綁定，會在引擎層級觸發 `Segmentation fault`。該測試已設為 Skip，不影響一般分散式架構下的正常推理。

### 3. 區域網路自動尋找測試 (`test_discovery.py`)
- ✅ `test_register_and_unregister`: 廣播器能成功註冊與取消 mDNS 服務。
- ✅ `test_listener_discovers_announcer`: 監聽器能自動偵測到新上線的節點廣播。
- ✅ `test_nodes_sorted_by_start_layer`: 多個節點能依據 `start_layer` 正確排序。
- ✅ `test_node_removal_on_unregister`: 節點取消廣播後，監聽器會自動將其從可用清單移除。
- ✅ `test_manual_mode_uses_node_urls`: 手動模式能向後相容讀取 `NODE_URLS` 環境變數。

> **最終執行結果**: `23 passed, 1 skipped in 14.04s` (核心邏輯與連線機制全數通過)

## 設計決策

| 元件 | 技術選擇 | 理由 |
|------|---------|------|
| 推理引擎 | MLX | 原生支援 Apple Silicon Unified Memory |
| 快取格式 | safetensors | 快速零拷貝讀寫，安全性佳 |
| 背景 I/O | Python threading + Queue | 解放主執行緒，不阻斷 MLX 計算圖 |
| 網路 API | FastAPI + uvicorn | 非同步高效能 |
| 資料序列化 | msgpack + numpy | 解決 JSON 傳輸 Tensor 效能低落問題，傳輸速度提升 ~10 倍 |
| LRU 機制 | OrderedDict | Python 內建，O(1) 更新順序 |

## 效能測試 (Performance Benchmarks)

原本使用 HTTP JSON 陣列傳輸 Tensor 時，網路負載與反序列化時間極高。經由匯入 `msgpack` 並直接轉換 `numpy` bytes 傳輸後，推論延遲大幅度縮減：

*   **優化前 (JSON Serialization)**: 第 1 個 Block 耗時 ~221ms。
*   **優化後 (`msgpack` Binary Serialization)**: 第 1 個 Block 耗時 **~23ms**，後續 Block 耗時縮減至 **~8ms**，整體效能提升約 10 倍。

## 已知限制（PoC 階段）

- 本專案目前使用簡單的 HTTP 傳輸。在跨實體機器的生產環境中，可考慮改用 Thunderbolt RPC 或 gRPC 甚至更底層的通訊協定。
- 目前放棄了背景執行緒中呼叫 `mx.load()` 進行特徵預取 (Prefetch) 的機制，因 MLX 引擎尚未支援跨執行緒與主執行緒交錯求值的激烈並行（會引發 Segmentation fault）。目前依賴作業系統的 Page Cache 提供基礎預取效能。


## 參考專案

- [exo](https://github.com/exo-explore/exo) - 多設備 LLM 叢集框架
- [omlx](https://github.com/Seb-sti1/omlx) - MLX SSD KV Cache 卸載
