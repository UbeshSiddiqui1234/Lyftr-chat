"""Unexpected server errors must stay JSON and keep CORS headers."""

from fastapi.testclient import TestClient

from app.config import ProviderKey, Settings
from app.database import Database
from app.gateway import LIVE_INFERENCE, Completion, Gateway
from app.main import create_app


def test_unexpected_error_is_json_and_stores_nothing(tmp_path, monkeypatch):
    settings = Settings(
        database_path=str(tmp_path / "app.db"),
        keys=(ProviderKey(id="key_1", model="openai/fake", api_key="sk-unused"),),
        cors_origins=("http://localhost:3000",),
        cache_enabled=False,
    )

    async def complete(self, messages, simulated=None):
        return Completion("hello", LIVE_INFERENCE, "openai/fake", 1, 1, "key_1", 0)

    def fail(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(Gateway, "complete", complete)
    monkeypatch.setattr(Database, "save_turn", fail)

    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post(
            "/chat",
            json={"session_id": "s-error", "message": "hi"},
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 500
        assert response.json() == {"detail": "internal server error"}
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "secret" not in response.text
        assert client.get("/sessions/s-error/messages").status_code == 404
