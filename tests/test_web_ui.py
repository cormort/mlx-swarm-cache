import os
import pytest
from fastapi.testclient import TestClient
from src.orchestrator.coordinator import app, MODELS_DIR

client = TestClient(app)

def test_web_ui_root_mount():
    """確保根目錄 / 能成功回傳 index.html 的內容"""
    # 如果 src/web 存在，根目錄應該要回傳 index.html
    if os.path.exists("src/web/index.html"):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MLX Swarm Cache" in response.text
    else:
        pytest.skip("src/web/index.html not found")

def test_list_models_empty():
    """測試 /v1/models 在沒有模型時回傳空陣列"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)

def test_list_models_with_mock_files(tmp_path, monkeypatch):
    """測試 /v1/models 在掃描到 config.json 時會回傳模型清單"""
    mock_models_dir = tmp_path / "mock_models"
    mock_models_dir.mkdir()
    
    # 建立一個測試用的模型結構
    repo_dir = mock_models_dir / "mlx-community" / "test-model"
    repo_dir.mkdir(parents=True)
    
    # 建立 config.json
    (repo_dir / "config.json").touch()
    
    # 替換 MODELS_DIR
    monkeypatch.setattr("src.orchestrator.coordinator.MODELS_DIR", str(mock_models_dir))
    
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["models"]) == 1
    assert "mlx-community/test-model" in data["models"][0]["repo_id"]
    
def test_download_model_endpoint():
    """測試 /v1/models/download 能正確回傳 downloading 狀態"""
    payload = {"repo_id": "test-org/dummy-repo"}
    response = client.post("/v1/models/download", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "downloading"
    assert data["repo_id"] == "test-org/dummy-repo"
    assert "test-org/dummy-repo" in data["local_dir"]
