"""End-to-end tests for the API contract, run against a real server process."""

import concurrent.futures
import uuid

import pytest

from conftest import KEY_1, KEY_2, Server, server_env


def sid() -> str:
    return f"s-{uuid.uuid4().hex}"


def text(label: str = "question") -> str:
    return f"{label} {uuid.uuid4().hex}"


def chat(api, session_id=None, message=None, failure=None, **extra):
    headers = {"X-Simulate-Failure": failure} if failure else {}
    body = {"session_id": session_id or sid(), "message": message or text(), **extra}
    return api.post("/chat", json=body, headers=headers)


def history(api, session_id):
    return api.get(f"/sessions/{session_id}/messages")


def test_health_shape(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_chat_response_contract(api):
    session_id = sid()
    response = chat(api, session_id)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["reply"]
    meta = data["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "openai/fake-model-a"
    assert isinstance(meta["latency_ms"], int) and meta["latency_ms"] > 0
    assert isinstance(meta["input_tokens"], int) and meta["input_tokens"] > 0
    assert meta["output_tokens"] == 7
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0


def test_multi_turn_sends_history_and_stores_it(api, provider):
    session_id = sid()
    first, second = text("first"), text("second")
    reply = chat(api, session_id, first).json()["reply"]
    chat(api, session_id, second)
    sent = provider.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert [m["content"] for m in sent[1:3]] == [first, reply]

    stored = history(api, session_id).json()
    assert stored["session_id"] == session_id
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"] * 2
    assert "metadata" not in stored["messages"][0]
    assert stored["messages"][1]["metadata"]["input_tokens"] > 0
    assert all(m["role"] != "system" for m in stored["messages"])


def test_same_first_message_in_new_session_hits_cache(api, provider):
    message = text()
    live = chat(api, message=message).json()
    calls = len(provider.calls)
    cached = chat(api, message=message).json()
    assert len(provider.calls) == calls
    assert cached["reply"] == live["reply"]
    meta = cached["metadata"]
    assert meta["resolution_source"] == "LLM_CACHE"
    assert meta["key_id"] is None
    assert meta["fallback_attempts"] == 0
    assert meta["input_tokens"] == live["metadata"]["input_tokens"]
    assert meta["output_tokens"] == live["metadata"]["output_tokens"]
    assert meta["model"] == live["metadata"]["model"]
    assert meta["latency_ms"] < live["metadata"]["latency_ms"]


def test_different_message_is_never_cached(api):
    message = text()
    chat(api, message=message)
    other = chat(api, message=message + "!").json()
    assert other["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_same_message_with_different_history_is_not_cached(api):
    repeated = text("repeat")
    chat(api, message=repeated)
    session_id = sid()
    chat(api, session_id, text("context"))
    response = chat(api, session_id, repeated).json()
    assert response["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_long_conversations_are_cached(api):
    turns = [text(f"turn{i}") for i in range(4)]
    first_session = sid()
    for turn in turns:
        chat(api, first_session, turn)
    second_session = sid()
    results = [chat(api, second_session, turn).json() for turn in turns]
    # Replies match turn by turn, so every prefix of the conversation is cached.
    assert all(r["metadata"]["resolution_source"] == "LLM_CACHE" for r in results)


def test_cache_shared_across_models_after_failover(api, provider):
    message = text()
    provider.failures[KEY_1] = 429
    first = chat(api, message=message).json()
    assert first["metadata"]["key_id"] == "key_2"
    provider.failures.clear()
    second = chat(api, message=message).json()
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["metadata"]["model"] == "openai/fake-model-b"
    assert second["reply"] == first["reply"]


def test_tokens_are_per_request(api):
    a = chat(api, message="a" * 40 + uuid.uuid4().hex).json()["metadata"]
    b = chat(api, message="b" * 40 + uuid.uuid4().hex).json()["metadata"]
    assert a["input_tokens"] == b["input_tokens"]
    session_id = sid()
    first = chat(api, session_id).json()["metadata"]
    second = chat(api, session_id).json()["metadata"]
    assert second["output_tokens"] == first["output_tokens"]


def test_system_prompt_overrides_and_persists(api, provider):
    session_id = sid()
    prompt = text("You are a pirate")
    chat(api, session_id, system_prompt=prompt)
    assert provider.calls[-1]["messages"][0] == {"role": "system", "content": prompt}
    chat(api, session_id)
    assert provider.calls[-1]["messages"][0]["content"] == prompt
    stored = history(api, session_id).json()["messages"]
    assert all(prompt not in m["content"] for m in stored)


def test_system_prompt_changes_cache_key(api):
    message = text()
    chat(api, message=message)
    other = chat(api, message=message, system_prompt=text("different")).json()
    assert other["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_failed_request_does_not_persist_system_prompt(api, provider):
    session_id = sid()
    response = chat(api, session_id, failure="*=429", system_prompt="should not stick")
    assert response.status_code == 503
    chat(api, session_id)
    assert provider.calls[-1]["messages"][0]["content"] != "should not stick"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"session_id": "s-1"},
        {"message": "hello"},
        {"session_id": "", "message": "hello"},
        {"session_id": "   ", "message": "hello"},
        {"session_id": "s" * 129, "message": "hello"},
        {"session_id": "s-1", "message": ""},
        {"session_id": "s-1", "message": "   \n\t "},
        {"session_id": "s-1", "message": "x" * 4001},
        {"session_id": 123, "message": "hello"},
        {"session_id": "s-1", "message": 123},
        {"session_id": "s-1", "message": "hello", "system_prompt": 5},
    ],
)
def test_invalid_bodies_return_422(api, provider, body):
    response = api.post("/chat", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert provider.calls == []


@pytest.mark.parametrize("raw", [b"{not json", b"", b"[]", b'"text"'])
def test_malformed_json_returns_422(api, raw):
    response = api.post("/chat", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_message_length_is_measured_after_trimming(api):
    session_id = sid()
    message = "x" * 3968 + uuid.uuid4().hex
    response = chat(api, session_id, f"   {message}   ")
    assert response.status_code == 200
    assert history(api, session_id).json()["messages"][0]["content"] == message


def test_rejected_request_is_not_stored(api):
    session_id = sid()
    assert api.post("/chat", json={"session_id": session_id, "message": " "}).status_code == 422
    assert history(api, session_id).status_code == 404


def test_unknown_session_returns_404(api):
    response = history(api, sid())
    assert response.status_code == 404
    assert response.json() == {"detail": "session not found"}


def test_framework_errors_are_json(api):
    assert api.get("/nope").json()["detail"]
    assert api.get("/chat").status_code == 405
    assert api.get("/chat").json()["detail"]


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_simulated_key_1_failure_fails_over(api, provider, status):
    session_id = sid()
    response = chat(api, session_id, failure=f"key_1={status}")
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["key_id"] == "key_2"
    assert data["metadata"]["fallback_attempts"] == 1
    assert data["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert data["reply"].startswith("answer")
    assert {call["key"] for call in provider.calls} == {KEY_2}


@pytest.mark.parametrize("header", ["*=429", "key_1=429,key_2=500", "*=503"])
def test_all_keys_simulated_returns_503(api, provider, header):
    session_id = sid()
    response = chat(api, session_id, failure=header)
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert provider.calls == []
    assert history(api, session_id).status_code == 404


def test_simulation_applies_to_one_request_only(api):
    session_id = sid()
    assert chat(api, session_id, failure="key_1=429").json()["metadata"]["key_id"] == "key_2"
    assert chat(api, session_id).json()["metadata"]["key_id"] == "key_1"


def test_simulation_takes_precedence_over_cache(api):
    message = text()
    chat(api, message=message)
    assert chat(api, message=message).json()["metadata"]["resolution_source"] == "LLM_CACHE"
    failover = chat(api, message=message, failure="key_1=429").json()["metadata"]
    assert failover["resolution_source"] == "LIVE_INFERENCE"
    assert failover["key_id"] == "key_2"
    assert chat(api, message=message, failure="*=429").status_code == 503


@pytest.mark.parametrize("header", ["garbage", "key_1", "key_1=404", "key_1=abc", "=429"])
def test_malformed_simulation_header_returns_422(api, provider, header):
    response = chat(api, failure=header)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert provider.calls == []


@pytest.mark.parametrize("status", [429, 500, 502, 503, 401, 403])
def test_real_provider_failure_fails_over(api, provider, status):
    provider.failures[KEY_1] = status
    response = chat(api)
    assert response.status_code == 200
    meta = response.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["model"] == "openai/fake-model-b"


def test_real_total_outage_returns_503_without_history(api, provider, server):
    provider.failures.update({KEY_1: 429, KEY_2: 500})
    session_id = sid()
    response = chat(api, session_id)
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert history(api, session_id).status_code == 404
    log = server.log_path.read_text(encoding="utf-8")
    assert KEY_1 not in log and KEY_2 not in log


def test_empty_provider_reply_fails_over_and_is_not_cached(api, provider):
    provider.empty_for.add(KEY_1)
    message = text()
    response = chat(api, message=message).json()
    assert response["reply"]
    assert response["metadata"]["key_id"] == "key_2"


def test_provider_timeout_fails_over(tmp_path, provider):
    server = Server(server_env(tmp_path, provider, LLM_TIMEOUT_SECONDS="0.05"))
    provider.delay = 0.3
    try:
        server.start()
        response = chat(server.client)
        assert response.status_code == 503
        assert response.json() == {"detail": "all providers unavailable"}
    finally:
        provider.delay = 0.08
        server.close()


def test_keys_never_appear_in_responses(api):
    session_id = sid()
    body = chat(api, session_id, failure="key_1=429").text
    body += history(api, session_id).text + api.get("/health").text
    assert KEY_1 not in body and KEY_2 not in body


def test_concurrent_requests_keep_session_order(api):
    session_id = sid()
    messages = [text(str(i)) for i in range(6)]
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        results = list(pool.map(lambda m: chat(api, session_id, m), messages))
    assert all(r.status_code == 200 for r in results)
    stored = history(api, session_id).json()["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant"] * 6


def test_cors_allows_frontend_and_simulation_header(api):
    response = api.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    error = api.post("/chat", json={}, headers={"Origin": "http://localhost:3000"})
    assert error.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_history_and_cache_survive_restart(tmp_path, provider):
    server = Server(server_env(tmp_path, provider))
    session_id, message = sid(), text()
    try:
        server.start()
        first = chat(server.client, session_id, message).json()
        server.stop()
        server.start()
        stored = history(server.client, session_id).json()["messages"]
        assert [m["content"] for m in stored] == [message, first["reply"]]
        again = chat(server.client, message=message).json()
        assert again["metadata"]["resolution_source"] == "LLM_CACHE"
        assert again["reply"] == first["reply"]
    finally:
        server.close()


def test_simulation_ignored_when_disabled(tmp_path, provider):
    server = Server(server_env(tmp_path, provider, ENABLE_FAILURE_SIMULATION="false"))
    try:
        server.start()
        response = chat(server.client, failure="*=429")
        assert response.status_code == 200
        assert response.json()["metadata"]["key_id"] == "key_1"
    finally:
        server.close()


def test_in_memory_cache_mode(tmp_path, provider):
    server = Server(server_env(tmp_path, provider, CACHE_TYPE="local"))
    message = text()
    try:
        server.start()
        chat(server.client, message=message)
        cached = chat(server.client, message=message).json()
        assert cached["metadata"]["resolution_source"] == "LLM_CACHE"
    finally:
        server.close()


def test_health_reports_missing_keys(tmp_path, provider):
    env = server_env(tmp_path, provider)
    for name in ("LLM_KEY_1", "LLM_KEY_2"):
        env.pop(name)
    server = Server(env)
    try:
        server.start()
        response = server.client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"detail": "no API keys configured"}
        assert chat(server.client).status_code == 503
    finally:
        server.close()
