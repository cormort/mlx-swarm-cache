import os
import subprocess
import sys
import time

import requests


def test_full_pipeline():
    print("🚀 Starting MLX-Swarm-Cache distributed test...")
    
    python_bin = sys.executable
    
    # 1. Start Worker 1
    env1 = os.environ.copy()
    env1["NODE_ID"] = "Worker_A"
    env1["PORT"] = "8011"
    w1 = subprocess.Popen([python_bin, "-m", "src.node.api_server"], env=env1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # 2. Start Worker 2
    env2 = os.environ.copy()
    env2["NODE_ID"] = "Worker_B"
    env2["PORT"] = "8012"
    w2 = subprocess.Popen([python_bin, "-m", "src.node.api_server"], env=env2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # 3. Start Coordinator
    env_c = os.environ.copy()
    env_c["DISCOVERY_MODE"] = "manual"
    env_c["NODE_URLS"] = "http://localhost:8011/forward,http://localhost:8012/forward"
    env_c["COORDINATOR_PORT"] = "8088"
    env_c["API_KEY"] = ""  # blank out inherited api keys
    coord = subprocess.Popen([python_bin, "-m", "src.orchestrator.coordinator"], env=env_c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    try:
        print("⏳ Waiting for servers to start...")
        time.sleep(10) # 確保 uvicorn 完全啟動
        
        print("📦 Sending Load Request...")
        # 測試載入比較小的 1B 模型避免 OOM 或下載太久
        res = requests.post("http://localhost:8088/v1/models/load", json={"repo_id": "mlx-community/Llama-3.2-1B-Instruct-4bit"}, timeout=120)
        print("Load response:", res.text)
        assert res.status_code == 200
        
        print("💬 Sending Chat Request...")
        chat_req = {
            "messages": [{"role": "user", "content": "What is the capital of Taiwan, reply in 1 short sentence."}],
            "max_tokens": 20
        }
        res = requests.post("http://localhost:8088/v1/chat/completions", json=chat_req, timeout=120)
        print("Chat response:", res.text)
        assert res.status_code == 200
        content = res.json()["choices"][0]["message"]["content"]
        print(">>> Reply:", content)
        assert len(content) > 0
        
        print("🗑️ Sending Unload Request...")
        res = requests.post("http://localhost:8088/v1/models/unload", timeout=120)
        print("Unload response:", res.text)
        assert res.status_code == 200
        
        print("✅ End-to-End Test Passed!")
    except Exception as e:
        print("❌ Test failed:", str(e))
        import traceback
        traceback.print_exc()
        # 列印 log 幫助 debug
        print("\n--- Coordinator Log ---")
        coord.terminate()
        print(coord.communicate()[0].decode())
        print("--- Worker A Log ---")
        w1.terminate()
        print(w1.communicate()[0].decode())
    finally:
        print("🛑 Shutting down servers...")
        try:
            w1.terminate()
        except Exception:
            pass
        try:
            w2.terminate()
        except Exception:
            pass
        try:
            coord.terminate()
        except Exception:
            pass
        w1.wait()
        w2.wait()
        coord.wait()

if __name__ == "__main__":
    test_full_pipeline()
