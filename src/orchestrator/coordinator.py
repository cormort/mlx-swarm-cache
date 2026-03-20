"""
coordinator.py — 叢集指揮官

負責：
  1. 將使用者 Prompt 轉為初始 Embedding（特徵矩陣）
  2. 依序呼叫各 Worker 節點的 /forward API，完成接力推理
  3. 收集每個 Block 的最終輸出（實務上會接 LM Head 轉為文字 Token）

修正紀錄 v2
-----------
- [Bug #3] 節點清單硬編碼兩個，新增第三台機器需要改程式碼
  → NODE_URLS 從環境變數讀取逗號分隔的 URL 清單，
    generate_step() 用 loop 依序呼叫，節點數量不再受限。

使用方式：
    python -m src.orchestrator.coordinator

兩個節點（預設）：
    NODE_URLS=http://localhost:8000/forward,http://192.168.1.100:8001/forward

三個節點（或更多）：
    NODE_URLS=http://localhost:8000/forward,http://192.168.1.100:8001/forward,http://192.168.1.101:8002/forward
"""

import os
import time

import mlx.core as mx
import msgpack
import numpy as np
import requests

# ─────────────────────────────────────────────
# 叢集節點設定（Bug #3 修正）
# ─────────────────────────────────────────────

_default_urls = "http://localhost:8000/forward,http://192.168.1.100:8001/forward"
NODE_URLS: list[str] = [
    u.strip()
    for u in os.getenv("NODE_URLS", _default_urls).split(",")
    if u.strip()
]

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 120))


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
    將特徵矩陣序列化為 msgpack 並 POST 至指定 Worker 節點。

    Returns:
        (輸出特徵矩陣, 節點回報的計算耗時 ms)；失敗時回傳 (None, 0)。
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
        
        np_array = np.frombuffer(
            data["hidden_states_bytes"], dtype=np_dtype
        ).reshape(shape)
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



def generate_step(
    block_id: str, current_hidden_states: mx.array
) -> mx.array | None:
    """
    執行一次完整的前向傳播，依序穿越 NODE_URLS 中所有節點（Bug #3 修正）。

    Returns:
        最終輸出特徵矩陣；任一節點失敗則回傳 None。
    """
    step_start = time.time()
    states = current_hidden_states
    timings: list[float] = []

    for i, url in enumerate(NODE_URLS, start=1):
        print(f"\n🏃 [第{i}棒] 傳送資料至 Node {i} ({url})...")
        states, t = call_worker_node(url, block_id, states)
        if states is None:
            return None
        timings.append(t)

    total_time = (time.time() - step_start) * 1000
    timing_str = ", ".join(f"N{i+1}: {t:.2f} ms" for i, t in enumerate(timings))
    print(f"✨ {block_id} 推理完成！(總耗時: {total_time:.2f} ms | {timing_str})")
    return states


# ─────────────────────────────────────────────
# 主程式：對話生成迴圈
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 MLX-Swarm-Cache 指揮官節點已上線")
    for i, url in enumerate(NODE_URLS, start=1):
        print(f"   Node {i}: {url}")
    print("=" * 50)

    prompt = "請幫我寫一個結合 omlx 與 exo 的架構..."
    current_states = text_to_embeddings(prompt)

    for i in range(1, 4):
        block_name = f"Token_Block_{i}"
        current_states = generate_step(block_name, current_states)
        if current_states is None:
            print("🚨 叢集運算中斷，請檢查節點狀態。")
            break
        print(f"-> 已產生第 {i} 個 Token Block")
