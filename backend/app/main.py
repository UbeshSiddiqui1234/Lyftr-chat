"""FastAPI application exposing /health, /chat and session history."""

import asyncio
import logging
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, load_settings
from .database import Database, utc_now
from .gateway import AllProvidersFailed, Gateway, parse_simulation_header
from .schemas import ChatMetadata, ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("app")


class JsonErrorMiddleware:
    """Turn unexpected exceptions into JSON while CORS headers still apply."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracked_send(message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception as exc:
            logger.error("unhandled error path=%s error=%s", scope.get("path"), type(exc).__name__)
            if response_started:
                raise
            response = JSONResponse(status_code=500, content={"detail": "internal server error"})
            await response(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    database = Database(settings.database_path)
    gateway = Gateway(settings)
    session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(database.init_schema)
        if len(settings.keys) < 2:
            logger.warning("key pool has %d key(s); failover needs at least 2", len(settings.keys))
        yield

    app = FastAPI(title="LLM Chat Gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(JsonErrorMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Simulate-Failure"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [
            {"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            await asyncio.to_thread(database.ping)
        except Exception as exc:
            logger.error("database health check failed error=%s", type(exc).__name__)
            return JSONResponse(status_code=503, content={"detail": "database unavailable"})
        if not gateway.keys:
            return JSONResponse(status_code=503, content={"detail": "no API keys configured"})
        return JSONResponse(
            {
                "status": "ok",
                "keys_configured": len(gateway.keys),
                "cache_enabled": gateway.cache_enabled,
            }
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        x_simulate_failure: str | None = Header(default=None),
    ):
        started = time.perf_counter()
        received_at = utc_now()

        simulated: dict[str, int] = {}
        if settings.enable_failure_simulation:
            try:
                simulated = parse_simulation_header(x_simulate_failure)
            except ValueError as exc:
                return JSONResponse(
                    status_code=422, content={"detail": f"invalid X-Simulate-Failure: {exc}"}
                )

        lock = session_locks.get(body.session_id)
        if lock is None:
            lock = asyncio.Lock()
            session_locks[body.session_id] = lock

        # One request per session at a time keeps user/assistant turns in order.
        async with lock:
            stored_prompt, history = await asyncio.to_thread(
                database.load_context, body.session_id, settings.max_history_messages
            )
            update_prompt = body.system_prompt is not None
            session_prompt = (body.system_prompt or None) if update_prompt else stored_prompt
            messages = [
                {"role": "system", "content": session_prompt or settings.default_system_prompt},
                *history,
                {"role": "user", "content": body.message},
            ]

            try:
                result = await gateway.complete(messages, simulated)
            except AllProvidersFailed:
                return JSONResponse(
                    status_code=503, content={"detail": "all providers unavailable"}
                )

            metadata = ChatMetadata(
                resolution_source=result.resolution_source,
                model=result.model,
                latency_ms=max(1, round((time.perf_counter() - started) * 1000)),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                key_id=result.key_id,
                fallback_attempts=result.fallback_attempts,
            )
            await asyncio.to_thread(
                database.save_turn,
                body.session_id,
                body.message,
                result.reply,
                metadata.model_dump(),
                received_at,
                session_prompt,
                update_prompt,
            )

        return ChatResponse(session_id=body.session_id, reply=result.reply, metadata=metadata)

    @app.get("/sessions/{session_id:path}/messages")
    async def session_messages(session_id: str) -> JSONResponse:
        messages = await asyncio.to_thread(database.get_messages, session_id)
        if messages is None:
            return JSONResponse(status_code=404, content={"detail": "session not found"})
        return JSONResponse({"session_id": session_id, "messages": messages})

    return app


app = create_app()
