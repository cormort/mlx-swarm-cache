#!/bin/bash

# 切換到腳本所在目錄，確保能找到 src 等目錄
cd "$(dirname "$0")"

# 設定終端機標題
echo -n -e "\033]0;MLX Swarm Cache 啟動程式\007"

echo "======================================================="
echo "🐝 MLX Swarm Cache 本地測試叢集啟動程式"
echo "   ✨ Auto-Discovery 模式：節點自動偵測"
echo "======================================================="

# 檢查虛擬環境是否存在
if [ ! -d ".venv" ]; then
    echo "⚠️ 找不到 .venv 目錄！"
    echo "請先在終端機執行以下指令建立環境："
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    echo "======================================================="
    read -p "請按任意鍵結束..." -n1 -s
    echo ""
    exit 1
fi

# 啟用虛擬環境
source .venv/bin/activate

# 建立清理函式：當腳本被終止時，強制砍掉所有子行程
cleanup() {
    echo ""
    echo "🛑 正在關閉所有背景服務..."
    # 傳送 SIGTERM 給目前 process group 下的所有程序
    kill 0
    exit 0
}

# 捕捉中斷訊號 (Ctrl+C 或是終端機關閉)
trap cleanup SIGINT SIGTERM EXIT

echo "🚀 [1/3] 啟動 Worker Node 1 (負責 0-16 層, Port: 8000)..."
export NODE_ID="mac_mini_m4_local"
export START_LAYER=0
export END_LAYER=16
export PORT=8000
.venv/bin/python -m src.node.api_server &

echo "🚀 [2/3] 啟動 Worker Node 2 (負責 16-32 層, Port: 8001)..."
export NODE_ID="macbook_air_local"
export START_LAYER=16
export END_LAYER=32
export PORT=8001
.venv/bin/python -m src.node.api_server &

echo "⏳ 等待 Worker 節點初始化與 mDNS 廣播 (5 秒)..."
sleep 5

echo "🚀 [3/3] 啟動 Coordinator (API Gateway, Port: 8080)..."
# Auto-Discovery 模式：不需要手動設定 NODE_URLS！
# Coordinator 會透過 mDNS 自動偵測上面兩個 Worker 節點
export DISCOVERY_MODE=auto
export COORDINATOR_PORT=8080
export API_KEY="sk-mlx-local" # 預設測試用的 API Key
.venv/bin/python -m src.orchestrator.coordinator &

echo "======================================================="
echo "✅ 本地測試叢集已成功啟動並在背景運行中！"
echo ""
echo "📍 服務端點："
echo "   - Node 1:      http://localhost:8000"
echo "   - Node 2:      http://localhost:8001"
echo "   - Coordinator: http://localhost:8080"
echo "   - 🌐 Web UI:   http://localhost:8080"
echo ""
echo "🔍 Auto-Discovery 已啟用！節點會自動透過 mDNS 被發現。"
echo ""
echo "🔑 測試用 API Key: sk-mlx-local"
echo "   (請在 Web UI 右上角的 API Key 欄位輸入)"
echo ""
echo "💡 查看已發現的節點："
echo "   curl http://localhost:8080/v1/nodes"
echo ""
echo "💡 API 呼叫範例:"
echo "   curl -X POST \"http://localhost:8080/v1/chat/completions\" \\"
echo "        -H \"Content-Type: application/json\" \\"
echo "        -H \"Authorization: Bearer sk-mlx-local\" \\"
echo "        -d '{\"model\": \"mlx-swarm-cache\", \"messages\": [{\"role\": \"user\", \"content\": \"你好！\"}]}'"
echo "======================================================="
echo "🔴 保持此視窗開啟以維持服務執行。"
echo "🔴 關閉此視窗或按下 Ctrl+C 即可一次終止所有背景服務。"

# 等待背景程序執行 (卡住以防腳本結束)
wait
