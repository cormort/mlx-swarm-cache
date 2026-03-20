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

import os
import time

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.node.worker_core import ExoWorkerNode

# ─────────────────────────────────────────────
# 節點全域狀態（從環境變數讀取）
# ─────────────────────────────────────────────

NODE_ID = os.getenv("NODE_ID", "default_node")
START_LAYER = int(os.getenv("START_LAYER", 0))
END_LAYER = int(os.getenv("END_LAYER", 16))
PORT = int(os.getenv("PORT", 8000))

worker = ExoWorkerNode(
    node_id=NODE_ID,
    assigned_layers=range(START_LAYER, END_LAYER),
    max_ram_blocks=5,
)

# ─────────────────────────────────────────────
# FastAPI 應用與資料模型
# ─────────────────────────────────────────────

app = FastAPI(
    title="MLX-Swarm-Cache Worker",
    description=f"Worker node [{NODE_ID}] handling layers {START_LAYER}-{END_LAYER - 1}",
    version="0.1.0",
)


class ForwardRequest(BaseModel):
    block_id: str
    # Tensor 序列化為多維 List 以便 JSON 傳輸
    hidden_states_list: list


class ForwardResponse(BaseModel):
    block_id: str
    hidden_states_list: list
    compute_time_ms: float


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


@app.post("/forward", response_model=ForwardResponse)
async def forward_pass(req: ForwardRequest):
    """
    接收上游特徵矩陣，執行本節點負責的層計算，回傳輸出特徵矩陣。
    """
    start_time = time.time()
    print(f"\n[{NODE_ID}] 🌐 收到 API 請求，處理 Block: {req.block_id}")

    try:
        # 1. 反序列化：JSON List → MLX Tensor
        hidden_states_tensor = mx.array(req.hidden_states_list)

        # 2. 執行推理（含自動 SSD KV Cache 管理）
        output_tensor = worker.forward_pass(hidden_states_tensor, req.block_id)

        # 3. 序列化：MLX Tensor → Python List
        output_list = output_tensor.tolist()

        compute_time_ms = (time.time() - start_time) * 1000
        print(f"[{NODE_ID}] ✅ 處理完成，耗時 {compute_time_ms:.2f} ms")

        return ForwardResponse(
            block_id=req.block_id,
            hidden_states_list=output_list,
            compute_time_ms=compute_time_ms,
        )

    except Exception as e:
        print(f"[{NODE_ID}] ❌ 處理失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# 程式進入點
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🟢 啟動 Worker 節點 [{NODE_ID}]，監聽 port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
