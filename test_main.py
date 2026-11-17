"""API Tests"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

FILE_NAME = "sunflower.jpg"


def test_read_main():
    """Tests that POST /v1/detect returns 200 OK with valid request body"""
    with open(FILE_NAME, "rb") as file:
        response = client.post(
            "/v1/detect",
            files={"file": (FILE_NAME, file, "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json() == {
        "file_name": FILE_NAME,
        "is_nsfw": False,
        "confidence_percentage": 100.0,
    }


def test_invalid_input():
    """Tests that POST /v1/detect returns 400 with empty request body"""
    response = client.post("/v1/detect", files={})
    assert response.status_code == 400


def test_repeated_requests_run_inference():
    """Both endpoints classify identical images on every request."""
    from unittest.mock import AsyncMock
    with open(FILE_NAME, "rb") as image:
        data = image.read()
    with patch("main.model", return_value=[{"label": "normal", "score": 1.0}]) as model:
        for _ in range(2):
            response = client.post("/v1/detect", files={"file": (FILE_NAME, data, "image/jpeg")})
            assert response.status_code == 200
        assert model.call_count == 2
        with patch("main.download_image", new_callable=AsyncMock, return_value=data) as download:
            response = client.post("/v1/detect/urls", json={"urls": ["https://example.com/image.jpg"] * 2})
            assert response.status_code == 200
            assert len(response.json()) == 2
            assert download.call_count == 2
        assert model.call_count == 4


def test_detect_urls():
    """Tests that POST /v1/detect/urls returns 200 OK with valid request body"""
    urls = [
        "https://raw.githubusercontent.com/steelcityamir/safe-content-ai/main/sunflower.jpg",
    ]
    response = client.post("/v1/detect/urls", json={"urls": urls})
    assert response.status_code == 200
    assert len(response.json()) == len(urls)
    assert response.json()[0] == {
        "url": urls[0],
        "is_nsfw": False,
        "confidence_percentage": 100.0,
    }
