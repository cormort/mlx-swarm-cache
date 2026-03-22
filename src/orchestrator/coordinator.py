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

import asyncio
import logging
import os
import pathlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import mlx.core as mx
import mlx.nn as nn
import msgpack
import numpy as np
import requests
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from src.discovery.listener import SwarmListener

logger = logging.getLogger("mlx-swarm")

# ─────────────────────────────────────────────
# 叢集節點設定（支援 Auto-Discovery 與手動模式）
# ─────────────────────────────────────────────

DISCOVERY_MODE = os.getenv("DISCOVERY_MODE", "auto").strip().lower()

# 手動模式：從環境變數讀取 NODE_URLS
_default_urls = "http://localhost:8000/forward,http://localhost:8001/forward"
_MANUAL_NODE_URLS: list[str] = [
    u.strip() for u in os.getenv("NODE_URLS", _default_urls).split(",") if u.strip()
]

# 自動模式：SwarmListener 實例延遲至 lifespan 建立，避免 import 副作用
_listener: SwarmListener | None = None

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 120))

class NetworkSwarmLayer(nn.Module):
    """代理網路模型的單一巨大神經網路層。
    
    將此層塞入 mlx_lm 模型架構中取代真實的 Transformer 層。
    當 mlx_lm 的生成迴圈嘗試計算下一層時，我們攔截張量，並透過網路
    將特徵發送給 Worker 接力計算，最後再接上本機的 LM Head 生成文字。
    """
    def __init__(self, session_id: str = "default_session"):
        super().__init__()
        self.session_id = session_id
        # 欺騙 mlx-lm 的 make_prompt_cache 等內部檢測方法
        self.use_sliding = False
        self.use_sliding_window = False
        self.is_linear = False

    def __call__(self, x, mask=None, cache=None):
        # 將 x 送入叢集進行接力傳播
        out = generate_step(self.session_id, x)
        if out is None:
            raise RuntimeError("分散式網路模型前向傳播失敗，叢集異常")
        
        # 欺騙 mlx-lm 的 KV Cache：
        # mlx-lm 的 generator 會呼叫 mx.eval() 評估 cache.state，並檢查 cache.keys.shape
        # 為了避免 AttributeError: 'NoneType' object has no attribute 'shape'，
        # 我們手動塞入極小且無意義的 tensor，以維護 offset 狀態，騙過迴圈。
        if cache is not None:
            seq_len = x.shape[1]
            step = getattr(cache, "step", 256)
            if getattr(cache, "keys", None) is None:
                cache.keys = mx.zeros((1, 1, seq_len + step, 1), dtype=x.dtype)
                cache.values = mx.zeros((1, 1, seq_len + step, 1), dtype=x.dtype)
                cache.offset = seq_len
            else:
                cache.offset += seq_len
                if cache.offset > cache.keys.shape[2]:
                    cache.keys = mx.zeros((1, 1, cache.offset + step, 1), dtype=x.dtype)
                    cache.values = mx.zeros((1, 1, cache.offset + step, 1), dtype=x.dtype)
                    
        return out

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


# 模型下載用的共用執行緒池
_download_executor = ThreadPoolExecutor(max_workers=2)

# ── 全域模型狀態（mlx-lm） ──────────────────────────────────
_loaded_model = None          # mlx_lm 的 model 物件
_loaded_tokenizer = None      # mlx_lm 的 tokenizer
_loaded_model_id: str | None = None  # 目前載入的 repo_id
_model_loading = False        # 是否正在載入中


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
    logger.info("✍️  使用者輸入: '%s'", prompt)
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
        logger.error("❌ 節點 %s 請求逾時（>%ss）", url, REQUEST_TIMEOUT)
        return None, 0.0
    except requests.exceptions.ConnectionError:
        logger.error("❌ 無法連線至節點 %s，請確認節點是否已啟動", url)
        return None, 0.0
    except requests.exceptions.RequestException as e:
        logger.error("❌ 呼叫節點 %s 失敗: %s", url, e)
        return None, 0.0
    except Exception as e:
        logger.error("❌ 解析節點 %s 回傳資料失敗: %s", url, e)
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
        logger.error("❌ 沒有可用的 Worker 節點！請確認節點已啟動。")
        return None

    step_start = time.time()
    states = current_hidden_states
    timings: list[float] = []

    for i, url in enumerate(node_urls, start=1):
        logger.info("🏃 [第%d棒] 傳送資料至 Node %d (%s)...", i, i, url)
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
    logger.info("✨ %s 推理完成！(總耗時: %.2f ms | %s)", block_id, total_time, timing_str)

    # 最終輸出的隱藏層狀態
    return states


# ─────────────────────────────────────────────
# 生命週期管理 (Lifespan)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _listener
    port = int(os.getenv("COORDINATOR_PORT", 8080))
    logger.info("=" * 50)
    logger.info("🚀 MLX-Swarm-Cache 指揮官 API Gateway 已啟動 (Port %d)", port)

    if DISCOVERY_MODE == "auto":
        _listener = SwarmListener()
        logger.info("   模式: 🔍 Auto-Discovery (mDNS/Zeroconf)")
        # ❗ Zeroconf 內部會在 event loop 上排程，用 executor 避免 EventLoopBlocked
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _listener.start)
        logger.info("   等待 Worker 節點自動上線...")
    else:
        logger.info("   模式: 📋 Manual (NODE_URLS)")
        for i, url in enumerate(_MANUAL_NODE_URLS, start=1):
            logger.info("   Node %d: %s", i, url)
    logger.info("=" * 50)

    yield  # 交出控制權給 FastAPI 執行期間

    logger.info("🛑 Coordinator 正在關閉...")
    if _listener is not None:
        await asyncio.get_running_loop().run_in_executor(None, _listener.stop)


# ─────────────────────────────────────────────
# FastAPI OpenAI 相容 API 介面
# ─────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")

app = FastAPI(
    title="MLX-Swarm-Cache Coordinator API",
    description="OpenAI-compatible API Gateway for the MLX distributed inference swarm.",
    lifespan=lifespan,
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


# ─────────────────────────────────────────────
# 模型管理 API (/v1/models)
# ─────────────────────────────────────────────

MODELS_DIR = os.getenv("MODELS_DIR", "./models")


class DownloadRequest(BaseModel):
    repo_id: str


class LoadRequest(BaseModel):
    repo_id: str


@app.get("/v1/models")
async def list_models():
    """掃描本機下載儲存資料夾，列出已下載的 Hugging Face 模型清單。"""
    if not os.path.exists(MODELS_DIR):
        return {"models": []}

    models = []
    for root, dirs, files in os.walk(MODELS_DIR):
        if "config.json" in files:
            rel_path = os.path.relpath(root, MODELS_DIR)
            models.append({
                "repo_id": rel_path,
                "local_path": root,
                "loaded": rel_path == _loaded_model_id,
            })

    return {"models": models}


@app.get("/v1/models/status")
async def model_status():
    """查詢目前載入的模型狀態。"""
    return {
        "loaded": _loaded_model_id is not None,
        "model_id": _loaded_model_id,
        "loading": _model_loading,
    }


@app.post("/v1/models/load")
async def load_model(req: LoadRequest):
    """載入指定模型到記憶體。

    使用 mlx-lm 的 load() 載入模型權重與 tokenizer。
    支援本地路徑或 HuggingFace repo_id。
    """
    global _loaded_model, _loaded_tokenizer, _loaded_model_id, _model_loading

    if _model_loading:
        return {"status": "error", "message": "模型正在載入中，請稍後再試"}

    repo_id = req.repo_id

    # 如果已載入同一模型，直接回傳
    if _loaded_model_id == repo_id:
        return {"status": "ok", "message": f"模型 {repo_id} 已經載入", "model_id": repo_id}

    # 先卸載舊模型
    if _loaded_model is not None:
        _loaded_model = None
        _loaded_tokenizer = None
        _loaded_model_id = None
        import gc; gc.collect()

    _model_loading = True
    try:
        # 優先查找本地下載的模型
        local_path = os.path.join(MODELS_DIR, repo_id)
        model_path = local_path if os.path.exists(local_path) else repo_id

        logger.info("📦 正在載入模型: %s", model_path)

        from mlx_lm import load as mlx_load

        loop = asyncio.get_running_loop()
        model, tokenizer = await loop.run_in_executor(
            None, lambda: mlx_load(model_path)
        )

        target_layers_attr = getattr(model, "model", model)
        if hasattr(target_layers_attr, "layers"):
            total_layers = len(target_layers_attr.layers)

            # 動態分配給 Worker
            if DISCOVERY_MODE == "auto" and _listener is not None:
                worker_base_urls = _listener.get_nodes_base_urls()
            else:
                worker_base_urls = [u.replace("/forward", "") for u in _MANUAL_NODE_URLS]

            node_count = len(worker_base_urls)
            if node_count > 0:
                layers_per_node = total_layers // node_count
                
                logger.info("📡 發現 %d 個 Worker 節點，準備佈署 %d 層網路...", node_count, total_layers)
                tasks = []
                for i, base_url in enumerate(worker_base_urls):
                    start = i * layers_per_node
                    end = total_layers if i == node_count - 1 else (i + 1) * layers_per_node
                    
                    # 平行送出載入要求
                    req_data = {"repo_id": repo_id, "start_layer": start, "end_layer": end}
                    tasks.append(
                        loop.run_in_executor(
                            None, 
                            lambda u=base_url, d=req_data: requests.post(
                                f"{u}/load", json=d, timeout=300
                            )
                        )
                    )
                
                # 等待所有 Worker 載入完成
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, Exception):
                        raise RuntimeError(f"節點載入失敗: {res}")
                    elif res.status_code != 200:
                        raise RuntimeError(f"節點載入失敗 ({res.status_code}): {res.text}")

                logger.info("✅ 所有 Worker 節點已載入負責的神經網路層，記憶體分配完成。")
                
                # 替換 Coordinator 的模型架構，拔除真實權重，並植入單一網路代理層
                try:
                    target_layers_attr.layers[:] = [NetworkSwarmLayer("shared_chat_session")]
                except AttributeError:
                    logger.warning("模型架構的 layers 可能是純 Tuple 或無法 inplace 修改，嘗試覆蓋隱藏屬性 _layers")
                    target_layers_attr._layers = [NetworkSwarmLayer("shared_chat_session")]
                import gc
                gc.collect()
            else:
                logger.warning("⚠️ 沒有發現任何 Worker 節點，將在 Coordinator 上執行單機推理。")
        else:
            logger.warning("⚠️ 該模型未偵測到標準 layers 結構，將嘗試單機推理。")

        _loaded_model = model
        _loaded_tokenizer = tokenizer
        _loaded_model_id = repo_id

        logger.info("✅ 模型 %s 載入完成！", repo_id)
        return {"status": "ok", "message": f"模型 {repo_id} 載入完成", "model_id": repo_id}

    except Exception as e:
        logger.error("❌ 模型載入失敗: %s", e)
        return {"status": "error", "message": f"模型載入失敗: {e}"}
    finally:
        _model_loading = False


@app.post("/v1/models/unload")
async def unload_model():
    """卸載目前模型，釋放記憶體。"""
    global _loaded_model, _loaded_tokenizer, _loaded_model_id

    if _loaded_model is None:
        return {"status": "ok", "message": "沒有載入中的模型"}

    old_id = _loaded_model_id
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_model_id = None

    import gc
    gc.collect()

    # 發送 Unload 給所有 Worker
    if DISCOVERY_MODE == "auto" and _listener is not None:
        worker_base_urls = _listener.get_nodes_base_urls()
    else:
        worker_base_urls = [u.replace("/forward", "") for u in _MANUAL_NODE_URLS]

    if worker_base_urls:
        def _unload_req(u):
            try:
                requests.post(f"{u}/unload", timeout=10)
            except Exception as e:
                logger.warning("⚠️ 通知節點 %s 卸載失敗: %s", u, e)
                
        loop = asyncio.get_running_loop()
        for u in worker_base_urls:
            asyncio.create_task(loop.run_in_executor(None, _unload_req, u))

    logger.info("🗑️ 模型 %s 已卸載，記憶體已釋放", old_id)
    return {"status": "ok", "message": f"模型 {old_id} 已卸載"}


@app.get("/v1/models/search")
async def search_models(q: str = "", limit: int = 20):
    """搜尋 HuggingFace Hub 上的 MLX 量化模型。

    透過 HuggingFace API 搜尋 mlx-community 等組織發布的量化模型，
    回傳模型名稱、下載次數與大小等資訊，方便使用者在 Web UI 上直接瀏覽與下載。

    Args:
        q: 搜尋關鍵字（例如 "llama", "qwen"）。留空則列出熱門模型。
        limit: 回傳筆數上限（預設 20）。
    """
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        # 搜尋 MLX 格式的模型（通常由 mlx-community 發布）
        search_query = f"mlx {q}" if q else "mlx"
        models = api.list_models(
            search=search_query,
            sort="downloads",
            limit=limit,
        )
        results = []
        for m in models:
            results.append({
                "repo_id": m.id,
                "downloads": m.downloads,
                "likes": m.likes,
                "last_modified": str(m.last_modified) if m.last_modified else None,
            })
        return {"models": results, "query": q}
    except Exception as e:
        logger.error("❌ 搜尋 HuggingFace 模型失敗: %s", e)
        return {"models": [], "query": q, "error": str(e)}


@app.post("/v1/models/download")
async def download_model(req: DownloadRequest):
    """
    接收像是 mlx-community/Llama-3.2-1B-Instruct-4bit 這樣的 repo_id，
    透過 huggingface_hub 啟動背景下載。
    """
    repo_id = req.repo_id
    local_dir = os.path.join(MODELS_DIR, repo_id)
    
    def _do_download():
        logger.info("📥 開始從 HuggingFace 下載模型: %s -> %s", repo_id, local_dir)
        try:
            snapshot_download(repo_id, local_dir=local_dir)
            logger.info("✅ 模型 %s 下載完成", repo_id)
        except Exception as e:
            logger.error("❌ 模型 %s 下載失敗: %s", repo_id, e)

    async def _bg_download():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_download_executor, _do_download)

    task = asyncio.create_task(_bg_download())
    task.add_done_callback(lambda t: t.result() if not t.cancelled() else None)
    return {"status": "downloading", "repo_id": repo_id, "local_dir": local_dir}


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
    """接收 OpenAI 格式的對話請求，使用 mlx-lm 進行真實推理。"""
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    if _loaded_model is None or _loaded_tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="尚未載入模型。請先透過 /v1/models/load 載入模型。",
        )

    prompt = req.messages[-1].content
    logger.info("🔗 [Gateway] 收到 API 請求: '%s' (模型: %s)", prompt, _loaded_model_id)

    try:
        from mlx_lm import generate as mlx_generate

        # 建構對話格式（如果 tokenizer 支援 chat template）
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if hasattr(_loaded_tokenizer, "apply_chat_template"):
            formatted_prompt = _loaded_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        max_tokens = min(req.max_tokens, 2048)
        start_time = time.time()
        
        session_id = f"chatcmpl-{uuid.uuid4().hex}"

        # 如果我們是分散式架構下的 NetworkSwarmLayer，動態注入 session_id 來分離對話序列的 Cache
        target_layers_attr = getattr(_loaded_model, "model", _loaded_model)
        if hasattr(target_layers_attr, "layers") and target_layers_attr.layers:
            if isinstance(target_layers_attr.layers[0], NetworkSwarmLayer):
                target_layers_attr.layers[0].session_id = session_id

        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(
            None,
            lambda: mlx_generate(
                _loaded_model,
                _loaded_tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                verbose=False,
            ),
        )

        gen_time = time.time() - start_time
        # 粗略估算 token 數
        prompt_tokens = len(formatted_prompt.split())
        completion_tokens = len(reply.split())

        logger.info(
            "✨ 推理完成！(耗時: %.2f s, 產生 %d tokens)",
            gen_time, completion_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=_loaded_model_id or req.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except Exception as e:
        import traceback
        full_trace = traceback.format_exc()
        logger.error("❌ 推理失敗: %s\n%s", e, full_trace)
        raise HTTPException(status_code=500, detail=f"推理失敗: {e}") from e

# 掛載網頁管理前端 (必須在所有 API 靜態路由宣告之後)
_WEB_DIR = pathlib.Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    port = int(os.getenv("COORDINATOR_PORT", 8080))
    # 透過字串匯入可讓 uvicorn 平滑處理 worker 行程
    uvicorn.run("src.orchestrator.coordinator:app", host="0.0.0.0", port=port)
