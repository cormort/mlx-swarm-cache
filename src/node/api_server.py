"""
api_server.py — Worker FastAPI 服務（步驟四）

透過環境變數決定節點身份，暴露 POST /forward 端點接收特徵矩陣，
執行本節點負責的神經網路層計算後回傳結果。

啟動方式：
    export NODE_ID="mac_mini_m4"
    export START_LAYER=0
    export END_LAYER=16
    export PORT=8000
    python -m src.node.api_server

⚠️  PoC 限制：Tensor 以 JSON List 傳輸，效能瓶頸明顯。
    正式環境應改用 Thunderbolt RPC + 二進位序列化（例如 numpy buffer 或 msgpack）。
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import mlx.core as mx

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

from src.discovery.announcer import SwarmAnnouncer
from src.node.worker_core import ExoWorkerNode

logger = logging.getLogger("mlx-swarm")

# ─────────────────────────────────────────────
# 節點全域狀態（從環境變數讀取）
# ─────────────────────────────────────────────

NODE_ID = os.getenv("NODE_ID", "default_node")
START_LAYER = int(os.getenv("START_LAYER", 0))
END_LAYER = int(os.getenv("END_LAYER", 16))
PORT = int(os.getenv("PORT", 8000))

# Worker 與廣播器延遲至 lifespan 再初始化，避免 import 副作用
worker: ExoWorkerNode | None = None
announcer: SwarmAnnouncer | None = None

# ─────────────────────────────────────────────
# 生命週期管理 (Lifespan)
# ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker, announcer
    # --- Server 啟動前執行 ---
    worker = ExoWorkerNode(
        node_id=NODE_ID,
        assigned_layers=range(START_LAYER, END_LAYER),
        max_ram_blocks=5,
    )
    announcer = SwarmAnnouncer(
        node_id=NODE_ID,
        port=PORT,
        start_layer=START_LAYER,
        end_layer=END_LAYER,
    )
    logger.info("🟢 啟動 Worker 節點 [%s]，準備廣播 mDNS...", NODE_ID)
    announcer.register()

    yield  # 將控制權交給 FastAPI，開始處理 Request

    # --- Server 關閉時執行 ---
    logger.info("🛑 [%s] 正在關閉，移除 mDNS 廣播...", NODE_ID)
    announcer.unregister()
    if worker:
        worker.shutdown()


# ─────────────────────────────────────────────
# FastAPI 應用與資料模型
# ─────────────────────────────────────────────

app = FastAPI(
    title="MLX-Swarm-Cache Worker",
    description=(
        f"Worker node [{NODE_ID}] handling layers {START_LAYER}-{END_LAYER - 1}"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# 移除 Pydantic 模型，改用 Request/Response 處理 raw bytes

# ─────────────────────────────────────────────
# 端點
# ─────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """健康檢查：Coordinator 可用此端點確認節點是否上線。"""
    return {
        "status": "ok",
        "node_id": NODE_ID,
        "layers": f"{START_LAYER}-{END_LAYER - 1}",
    }


@app.post("/forward")
async def forward_pass(request: Request):
    """
    接收 msgpack 格式的特徵矩陣，執行本節點負責的層計算，最後回傳更新後的特徵矩陣。

    這個端點作為分散式推論網路中單一節點的進入點：
    1. 從 HTTP Body 提取二進位的 msgpack 資料。
    2. 反序列化成 NumPy 陣列，並轉換為 `mlx.core.array` 張量。
    3. 呼叫 `WorkerCore` 執行神經網路的前向傳播 (Forward Pass) 與 KV Cache 快取管理。
    4. 將計算結果再次序列化為 bytes 並封裝為 msgpack 回傳給 Coordinator (上一棒)。

    Args:
        request (Request): FastAPI 的原始 Request 物件，包含二進位的 Payload。

    Returns:
        Response: 包裝著 msgpack 序列化資料的原始 HTTP Response。
    """
    start_time = time.time()

    try:
        raw_body = await request.body()
        data = msgpack.unpackb(raw_body, raw=False)
        block_id = data["block_id"]

        logger.info("[%s] 🌐 收到 API 請求，處理 Block: %s", NODE_ID, block_id)

        # 1. 反序列化：bytes -> numpy -> MLX Tensor
        # msgpack 經常會把 bytes 轉成 bytes object, 我們透過 np.frombuffer 轉回數值陣列
        shape = tuple(data["shape"])
        dtype_str = data["dtype"]
        np_dtype = np.dtype(dtype_str)

        np_array = np.frombuffer(data["hidden_states_bytes"], dtype=np_dtype).reshape(
            shape
        )
        hidden_states_tensor = mx.array(np_array)

        # 2. 執行推理（含自動 SSD KV Cache 管理）
        output_tensor = worker.forward_pass(hidden_states_tensor, block_id)  # type: ignore[union-attr]

        # 3. 序列化：MLX Tensor -> numpy -> bytes
        # 注意 MLX tensor 先轉 numpy 才能取得底層 bytes
        out_np = np.array(output_tensor)
        out_bytes = out_np.tobytes()

        compute_time_ms = (time.time() - start_time) * 1000
        logger.info("[%s] ✅ 處理完成，耗時 %.2f ms", NODE_ID, compute_time_ms)

        response_data = {
            "block_id": block_id,
            "hidden_states_bytes": out_bytes,
            "shape": out_np.shape,
            "dtype": str(out_np.dtype),
            "compute_time_ms": compute_time_ms,
        }

        packed_response = msgpack.packb(response_data, use_bin_type=True)
        return Response(content=packed_response, media_type="application/msgpack")

    except Exception as e:
        logger.error("[%s] ❌ 處理失敗: %s", NODE_ID, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─────────────────────────────────────────────
# 程式進入點
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 注意：字串格式的 app 參考能讓 uvicorn 更好地管理 worker processes
    uvicorn.run("src.node.api_server:app", host="0.0.0.0", port=PORT)
