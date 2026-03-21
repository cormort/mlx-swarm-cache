"""
coordinator.py — 叢集指揮官

負責：
  1. 將使用者 Prompt 轉為初始 Embedding（特徵矩陣）
  2. 依序呼叫各 Worker 節點的 /forward API，完成接力推理
  3. 收集每個 Block 的最終輸出（實務上會接 LM Head 轉為文字 Token）

修正紀錄 v3
-----------
- [Bug #3] NODE_URLS 環境變數化，支援不限數量的節點。
- [Feature] Auto-Discovery：使用 mDNS/Zeroconf 自動偵測 Worker 節點，
  不再需要手動設定 NODE_URLS。透過 DISCOVERY_MODE 環境變數控制：
    - "auto"：（預設）自動偵測區域網路中的 Worker 節點
    - "manual"：使用 NODE_URLS 環境變數手動指定

使用方式：
    # 自動模式（預設，不需設定 NODE_URLS）
    python -m src.orchestrator.coordinator

    # 手動模式
    DISCOVERY_MODE=manual NODE_URLS=http://localhost:8000/forward,http://localhost:8001/forward \
      python -m src.orchestrator.coordinator
"""

import os
import time
import uuid

import mlx.core as mx
import msgpack
import numpy as np
import requests
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.discovery.listener import SwarmListener

# ─────────────────────────────────────────────
# 叢集節點設定（支援 Auto-Discovery 與手動模式）
# ─────────────────────────────────────────────

DISCOVERY_MODE = os.getenv("DISCOVERY_MODE", "auto").strip().lower()

# 手動模式：從環境變數讀取 NODE_URLS
_default_urls = "http://localhost:8000/forward,http://localhost:8001/forward"
_MANUAL_NODE_URLS: list[str] = [
    u.strip() for u in os.getenv("NODE_URLS", _default_urls).split(",") if u.strip()
]

# 自動模式：建立 mDNS 監聽器（在 __main__ 中啟動）
_listener: SwarmListener | None = None

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 120))


def get_active_node_urls() -> list[str]:
    """依當前模式取得排序過的 Worker 節點 URL 清單。

    - auto 模式：從 SwarmListener 動態取得（依 start_layer 排序）
    - manual 模式：回傳環境變數 NODE_URLS 的靜態清單

    Returns:
        現有可用節點的 Forward URL 清單。
    """
    if DISCOVERY_MODE == "auto" and _listener is not None:
        return _listener.get_node_urls()
    return _MANUAL_NODE_URLS


# ─────────────────────────────────────────────
# 輔助函式
# ─────────────────────────────────────────────


def text_to_embeddings(prompt: str) -> mx.array:
    """
    模擬 Tokenizer + Embedding 層。

    實務替換為：
        token_ids = tokenizer.encode(prompt)
        return embedding_table[token_ids]   # shape: (1, seq_len, dim)
    """
    print(f"✍️  使用者輸入: '{prompt}'")
    return mx.random.uniform(shape=(1, 16, 4096))


def call_worker_node(
    url: str, block_id: str, hidden_states: mx.array
) -> tuple[mx.array | None, float]:
    """
    將特徵矩陣序列化為 msgpack 並 POST 至指定 Worker 節點進行推理。

    此函式負責處理節點之間的網路通訊，使用 msgpack 二進位格式傳輸
    以取代 Pydantic/JSON，大幅降低網路頻寬負載與反序列化時間。

    Args:
        url (str): 目標 Worker 節點的 HTTP API 網址 (例如: http://localhost:8000/forward)。
        block_id (str): 當前處理的 Token Block 識別碼，用於追蹤工作進度或標註快取。
        hidden_states (mx.array): 準備要傳輸給 Worker 的 MLX 隱藏層特徵矩陣。

    Returns:
        tuple[mx.array | None, float]:
            - 成功時回傳 (計算完成的輸出特徵矩陣, 節點回報的計算耗時 ms)。
            - 失敗或發生網路錯誤時回傳 (None, 0.0)。
    """
    hs_np = np.array(hidden_states)
    payload_dict = {
        "block_id": block_id,
        "hidden_states_bytes": hs_np.tobytes(),
        "shape": hs_np.shape,
        "dtype": str(hs_np.dtype),
    }

    try:
        payload_bytes = msgpack.packb(payload_dict, use_bin_type=True)
        headers = {"Content-Type": "application/msgpack"}

        response = requests.post(
            url, data=payload_bytes, headers=headers, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        data = msgpack.unpackb(response.content, raw=False)

        shape = tuple(data["shape"])
        dtype_str = data["dtype"]
        np_dtype = np.dtype(dtype_str)

        np_array = np.frombuffer(data["hidden_states_bytes"], dtype=np_dtype).reshape(
            shape
        )
        output_tensor = mx.array(np_array)

        return output_tensor, data["compute_time_ms"]

    except requests.exceptions.Timeout:
        print(f"❌ 節點 {url} 請求逾時（>{REQUEST_TIMEOUT}s）")
        return None, 0.0
    except requests.exceptions.ConnectionError:
        print(f"❌ 無法連線至節點 {url}，請確認節點是否已啟動")
        return None, 0.0
    except requests.exceptions.RequestException as e:
        print(f"❌ 呼叫節點 {url} 失敗: {e}")
        return None, 0.0
    except Exception as e:
        print(f"❌ 解析節點 {url} 回傳資料失敗: {e}")
        return None, 0.0


def generate_step(block_id: str, current_hidden_states: mx.array) -> mx.array | None:
    """
    執行一次完整的前向傳播，依序穿越所有已發現的節點。

    在 Auto-Discovery 模式下，每次呼叫都會從 SwarmListener
    動態取得最新的節點清單，確保新上線的節點能即時加入推理。

    Returns:
        最終輸出特徵矩陣；任一節點失敗則回傳 None。
    """
    node_urls = get_active_node_urls()

    if not node_urls:
        print("❌ 沒有可用的 Worker 節點！請確認節點已啟動。")
        return None

    step_start = time.time()
    states = current_hidden_states
    timings: list[float] = []

    for i, url in enumerate(node_urls, start=1):
        print(f"\n🏃 [第{i}棒] 傳送資料至 Node {i} ({url})...")
        states, t = call_worker_node(url, block_id, states)
        if states is None:
            # 發生連線錯誤或 Timeout，如果是 Auto 模式則強制剔除殭屍節點
            if DISCOVERY_MODE == "auto" and _listener is not None:
                _listener.remove_node_by_url(url)
            # 任一節點斷線即中斷推理，確保不會產生錯誤的 Token
            return None
        timings.append(t)

    total_time = (time.time() - step_start) * 1000
    timing_str = ", ".join(f"N{i + 1}: {t:.2f} ms" for i, t in enumerate(timings))
    print(f"✨ {block_id} 推理完成！(總耗時: {total_time:.2f} ms | {timing_str})")

    # 最終輸出的隱藏層狀態
    return states


# ─────────────────────────────────────────────
# FastAPI OpenAI 相容 API 介面
# ─────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")

app = FastAPI(
    title="MLX-Swarm-Cache Coordinator API",
    description="OpenAI-compatible API Gateway for the MLX distributed inference swarm.",
)


@app.get("/v1/nodes")
async def list_nodes():
    """查看叢集中的節點狀態（支援 Auto-Discovery 與 Manual 模式）。"""
    if DISCOVERY_MODE == "auto" and _listener is not None:
        nodes = _listener.get_nodes_info()
        return {
            "mode": "auto",
            "node_count": len(nodes),
            "nodes": nodes,
        }
    return {
        "mode": "manual",
        "node_count": len(_MANUAL_NODE_URLS),
        "nodes": [{"forward_url": url} for url in _MANUAL_NODE_URLS],
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mlx-swarm-cache"
    messages: list[ChatMessage]
    max_tokens: int = 16


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


async def verify_api_key(authorization: str = Header(None)):
    """驗證 API_KEY，如果環境變數有設定 API_KEY，連線必須帶正確的 Bearer Token。"""
    if not API_KEY:
        return  # 若未設定環境變數 API_KEY，則不強制驗證
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    token = authorization.split("Bearer ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def chat_completions(req: ChatCompletionRequest):
    """
    接收 OpenAI 格式的對話請求，將其轉為特徵矩陣後丟入 Swarm 叢集中進行分散式推論。
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    prompt = req.messages[-1].content
    print(f"\n🔗 [Gateway] 收到外部 API 請求: '{prompt}'")

    current_states = text_to_embeddings(prompt)
    generated_blocks = 0

    # PoC 階段: 將 max_tokens 當作迴圈輪數來模擬長時間運算
    blocks_to_generate = min(req.max_tokens, 10)

    for i in range(1, blocks_to_generate + 1):
        block_name = f"Token_Block_{i}"
        current_states = generate_step(block_name, current_states)
        if current_states is None:
            raise HTTPException(
                status_code=500, detail="叢集運算中斷，請檢查 Worker 節點狀態"
            )
        generated_blocks += 1

    # 實務上這裡會將 current_states 通過 LM Head 轉換為文字，目前以模擬字串代替
    active_urls = get_active_node_urls()
    mock_reply = (
        f"這是 MLX-Swarm-Cache 產生的測試回應。\n"
        f"成功穿越了 {len(active_urls)} 個節點，經過 {generated_blocks} 輪推論接力完成！"
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=mock_reply),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(prompt),
            completion_tokens=generated_blocks,
            total_tokens=len(prompt) + generated_blocks,
        ),
    )


if __name__ == "__main__":
    import signal

    port = int(os.getenv("COORDINATOR_PORT", 8080))
    print("=" * 50)
    print(f"🚀 MLX-Swarm-Cache 指揮官 API Gateway 已啟動 (Port {port})")

    if DISCOVERY_MODE == "auto":
        _listener = SwarmListener()
        _listener.start()
        print("   模式: 🔍 Auto-Discovery (mDNS/Zeroconf)")
        print("   等待 Worker 節點自動上線...")
    else:
        print("   模式: 📋 Manual (NODE_URLS)")
        for i, url in enumerate(_MANUAL_NODE_URLS, start=1):
            print(f"   Node {i}: {url}")

    print("=" * 50)

    def _shutdown_handler(sig, frame):
        """捕捉關閉訊號，優雅清理 mDNS 監聽器。"""
        print("\n🛑 Coordinator 正在關閉...")
        if _listener is not None:
            _listener.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    uvicorn.run(app, host="0.0.0.0", port=port)
