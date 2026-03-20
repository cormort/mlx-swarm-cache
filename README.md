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

1. **Coordinator**: 接收使用者輸入，轉換特徵矩陣並排程。
2. **Worker Nodes**: 每個節點是一個獨立的 FastAPI 服務，負責特定神經網路層。
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
│   ├── node/
│   │   ├── worker_core.py          # MLX 推理與快取管理
│   │   └── api_server.py           # FastAPI 網路服務端點
│   └── orchestrator/
│       └── coordinator.py          # 指揮官：發送 Prompt 與網路排程
└── tests/
    └── test_cache_eviction.py      # 驗證 RAM 滿載時卸載行為
```

## 快速啟動 (Quickstart)

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 啟動 Node 1（例如 Mac mini M4，負責 Layer 0-15）

```bash
export NODE_ID="mac_mini_m4"
export START_LAYER=0
export END_LAYER=16
export PORT=8000
python -m src.node.api_server
```

### 啟動 Node 2（例如 MacBook Air，負責 Layer 16-31）

```bash
export NODE_ID="macbook_air"
export START_LAYER=16
export END_LAYER=32
export PORT=8001
python -m src.node.api_server
```

### 啟動指揮官（在 Node 1 或獨立機器上執行）

```bash
python -m src.orchestrator.coordinator
```

### 執行快取卸載測試

```bash
python -m pytest tests/test_cache_eviction.py -v
```

## 設計決策

| 元件 | 技術選擇 | 理由 |
|------|---------|------|
| 推理引擎 | MLX | 原生支援 Apple Silicon Unified Memory |
| 快取格式 | safetensors | 快速零拷貝讀寫，安全性佳 |
| 背景 I/O | Python threading + Queue | 解放主執行緒，不阻斷 MLX 計算圖 |
| 網路 API | FastAPI + uvicorn | 非同步高效能，Pydantic 型別驗證 |
| LRU 機制 | OrderedDict | Python 內建，O(1) 更新順序 |

## 已知限制（PoC 階段）

- HTTP JSON 傳輸 Tensor 效能差，正式環境應改用 Thunderbolt RPC + 二進位序列化
- 背景 I/O 執行緒寫入 `ssd_index` 存在 Race Condition，正式環境需加鎖
- 尚未實作 Prefetch 機制（預先從 SSD 暖機下一批 Block）

## 參考專案

- [exo](https://github.com/exo-explore/exo) - 多設備 LLM 叢集框架
- [omlx](https://github.com/Seb-sti1/omlx) - MLX SSD KV Cache 卸載
