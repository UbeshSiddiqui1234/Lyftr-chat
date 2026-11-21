# Gateway Chat

A multi-turn chat app with a LiteLLM gateway, API key failover, response caching and per-request metadata. History is stored in SQLite.

| Tier | Stack | Local URL |
| --- | --- | --- |
| Frontend | React, TypeScript, Vite, nginx | http://localhost:3000 |
| Backend | FastAPI, LiteLLM, SQLite | http://localhost:8000 |

## Run locally

1. Copy `.env.example` to `.env`.
2. Set `LLM_KEY_1` and `LLM_KEY_2`.
3. Start the stack:

```sh
make up
```

Stop it with `make down`. History stays in the `app-data` Docker volume.

On Windows without `make`, copy the env file yourself and run `docker compose up --build -d`.

## Configuration

All settings are environment variables, documented in `.env.example`.

Keys are tried in numeric order. Each key can use its own model or OpenAI-compatible endpoint through `LLM_MODEL_<n>` and `LLM_API_BASE_<n>`.

## How it behaves

- **Failover:** any provider failure on a key (429, quota, 5xx, auth error, timeout or empty reply) moves to the next key. If all keys fail, the API returns 503 and stores nothing.
- **Cache:** the key is the model plus the full message list. A hit is checked across every model in the pool, so a reply produced after failover is still found.
- **Failure simulation:** a request with `X-Simulate-Failure` skips the cache read, so the named keys are always exercised.
- **System prompt:** a value sent with `/chat` is remembered for that session. An empty string resets it to the default.

## Implementation notes

- **LiteLLM is pinned to 1.106.2**, the version its maintainers named as safe after the October 2026 supply-chain incident. All Python dependencies are locked by hash. Read their advisories before upgrading.
- **The LiteLLM `Cache` object is called directly** rather than through `acompletion(caching=True)`. In this version the automatic path crashes on a cache hit without the proxy extras, writes in the background so an immediate repeat can miss, and skips requests with more than four messages.
- **The default model is `gemini-3.5-flash-lite`** because `gemini-2.0-flash`, used in the spec's example, has been shut down.

## Tests

```sh
cd backend
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements-dev.txt
DATABASE_URL=sqlite:///./.pytest-import.db .venv/bin/python -m pytest tests
```

The suite runs a real server against a fake provider, so it needs no API keys.

To check a running backend, including a deployed one (this makes a few real model calls):

```sh
python scripts/smoke_test.py https://your-backend.example.com
```

## Deploy

`render.yaml` defines both services. Push the repo to GitHub, create a Render Blueprint from it, then set:

| Service | Variable | Value |
| --- | --- | --- |
| Backend | `LLM_KEY_1`, `LLM_KEY_2` | Your provider keys |
| Backend | `CORS_ORIGINS` | The frontend URL, with no trailing slash |
| Frontend | `VITE_API_BASE_URL` | The backend URL |

`VITE_API_BASE_URL` is compiled into the site, so redeploy the frontend after changing it.

The backend needs a persistent disk at `/data`, or history is lost on restart.
