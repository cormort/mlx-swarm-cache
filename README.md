# MLX-Swarm-Cache 🐝

MLX-Swarm-Cache 是一個實驗性的分散式 LLM 推理引擎概念驗證 (PoC) 專案。它專為 Apple Silicon 設計，旨在解決多台 Mac 設備協同運行大型語言模型時，單機統一記憶體 (Unified Memory) 容易耗盡的痛點。

本專案結合了分散式節點網路與非同步的 SSD KV 快取卸載機制，讓模型在跨設備推理時，也能優雅地管理記憶體。

## ✨ 核心特色

* **分散式推理 (Distributed Inference)**: 將大型神經網路模型切片，分配給多台 Mac 協同運算。
* **分層 KV 快取 (Tiered KV Caching)**: 當熱層 (RAM) 滿載時，透過非同步背景執行緒將最少使用的 Context 區塊以 `safetensors` 格式卸載至冷層 (SSD)。
* **無阻塞架構 (Non-blocking I/O)**: 確保磁碟寫入不會拖慢叢集節點間的網路傳輸與神經網路運算。

## 🙏 致謝與授權聲明 (Acknowledgments)

本專案是站在開源社群巨人的肩膀上所建立的微創新實驗。我們深表感謝並致敬以下兩個卓越的開源專案：

* [cite_start]**[exo](https://github.com/exo-explore/exo)**: 啟發了本專案將多台設備連結成 AI 叢集的分散式運算概念 [cite: 49]。本專案在工作節點的架構設計上參考了其理念。
* [cite_start]**[omlx](https://github.com/jundot/omlx)**: 提供了適用於 Apple Silicon 的連續批次處理與分層 KV 快取 (Tiered KV Caching) 的完美實作 [cite: 145, 146]。本專案的 SSD 卸載與 RAM 記憶體管理機制深受其程式碼啟發。

[cite_start]以上原專案皆採用 Apache-2.0 license 開源授權 [cite: 48, 209]。本專案亦在遵循 Apache 2.0 授權條款下發布。

## 🚀 快速啟動 (Quickstart)

假設您有兩台設備 (例如 Mac mini M4 與 MacBook Air) 準備進行叢集運算：

**1. 在節點 A (負責前半段網路層) 啟動 Worker:**
bash
export NODE_ID="mac_mini_m4"
export START_LAYER=0
export END_LAYER=16
export PORT=8000
python -m src.node.api_server

2. 在節點 B (負責後半段網路層) 啟動 Worker:
Bash
export NODE_ID="macbook_air"
export START_LAYER=16
export END_LAYER=32
export PORT=8001
python -m src.node.api_server

3. 啟動指揮官 (Coordinator) 進行推理:
Bash
python -m src.orchestrator.coordinator
