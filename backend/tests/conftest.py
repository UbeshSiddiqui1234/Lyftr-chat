"""Black-box fixtures: a real uvicorn server talking to a fake LLM provider."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
KEY_1 = "sk-test-secret-one"
KEY_2 = "sk-test-secret-two"


class FakeProvider:
    """OpenAI-compatible endpoint whose behavior is controlled per API key."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.failures: dict[str, int] = {}
        self.empty_for: set[str] = set()
        self.delay = 0.08
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                key = self.headers["authorization"].removeprefix("Bearer ")
                provider.calls.append({"key": key, "messages": body["messages"]})
                time.sleep(provider.delay)
                if key in provider.failures:
                    status = provider.failures[key]
                    self._send(status, {"error": {"message": f"upstream failure for {key}"}})
                    return
                text = "" if key in provider.empty_for else f"answer {len(provider.calls)}"
                chars = sum(len(m["content"]) for m in body["messages"])
                self._send(
                    200,
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": text},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": chars // 4 + 1,
                            "completion_tokens": 7,
                            "total_tokens": chars // 4 + 8,
                        },
                    },
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reset(self) -> None:
        self.failures.clear()
        self.empty_for.clear()
        self.calls.clear()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = Path(env["TEST_LOG_FILE"])
        self.process: subprocess.Popen | None = None
        self.client = httpx.Client(base_url=self.url, timeout=30)

    def start(self) -> None:
        log = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=BACKEND_DIR,
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if self.client.get("/health").status_code in (200, 503):
                    return
            except httpx.TransportError:
                time.sleep(0.2)
        raise RuntimeError(f"server did not start: {self.log_path.read_text()}")

    def stop(self) -> None:
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=15)
            self.process = None

    def close(self) -> None:
        self.stop()
        self.client.close()


def server_env(tmp: Path, provider: FakeProvider, **overrides: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LLM_", "CACHE_", "DATABASE_", "ENABLE_"))
    }
    env.update(
        DATABASE_URL=f"sqlite:///{(tmp / 'app.db').as_posix()}",
        CACHE_DIR=str(tmp / "cache"),
        CACHE_TYPE="disk",
        ENABLE_FAILURE_SIMULATION="true",
        CORS_ORIGINS="http://localhost:3000",
        LLM_KEY_1=KEY_1,
        LLM_MODEL_1="openai/fake-model-a",
        LLM_API_BASE_1=provider.base_url,
        LLM_KEY_2=KEY_2,
        LLM_MODEL_2="openai/fake-model-b",
        LLM_API_BASE_2=provider.base_url,
        LLM_TIMEOUT_SECONDS="5",
        TEST_LOG_FILE=str(tmp / "server.log"),
    )
    env.update(overrides)
    return env


@pytest.fixture(scope="session")
def provider():
    fake = FakeProvider()
    yield fake
    fake.close()


@pytest.fixture(scope="session")
def server(provider, tmp_path_factory):
    instance = Server(server_env(tmp_path_factory.mktemp("main"), provider))
    instance.start()
    yield instance
    instance.close()


@pytest.fixture(autouse=True)
def clean_provider(provider):
    provider.reset()


@pytest.fixture
def api(server):
    return server.client
