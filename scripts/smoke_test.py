"""Check a running backend against the API contract.

Usage: python scripts/smoke_test.py https://your-backend.example.com
Makes a handful of real model calls. Needs only the Python standard library.
"""

import json
import sys
import urllib.error
import urllib.request
import uuid


def request(base, method, path, body=None, headers=None):
    data = body if isinstance(body, bytes) else None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "User-Agent": "contract-smoke-test", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"non_json_body": raw[:120].decode(errors="replace")}


def main(base):
    base = base.rstrip("/")
    failures = []

    def check(name, passed, info=""):
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({info})" if info and not passed else ""))
        if not passed:
            failures.append(name)

    def chat(session, message, failure=None):
        headers = {"X-Simulate-Failure": failure} if failure else {}
        return request(base, "POST", "/chat", {"session_id": session, "message": message}, headers)

    def sid():
        return f"s-{uuid.uuid4().hex[:16]}"

    status, health = request(base, "GET", "/health")
    check("health is ok with 2+ keys", status == 200 and health.get("keys_configured", 0) >= 2, health)

    question = f"Reply with one short sentence about the number {uuid.uuid4().int % 100000}."
    first_session = sid()
    status, live = chat(first_session, question)
    live_meta = live.get("metadata", {})
    check("live reply", status == 200 and live_meta.get("resolution_source") == "LIVE_INFERENCE" and bool(live.get("reply")), live)
    check("tokens reported", live_meta.get("input_tokens", 0) > 0 and live_meta.get("output_tokens", 0) > 0, live_meta)

    status, cached = chat(sid(), question)
    cached_meta = cached.get("metadata", {})
    check("same message in new session hits cache", status == 200 and cached_meta.get("resolution_source") == "LLM_CACHE", cached_meta)
    check("cached reply is identical", cached.get("reply") == live.get("reply"))
    check("cache hit has null key and 0 fallbacks", cached_meta.get("key_id") is None and cached_meta.get("fallback_attempts") == 0, cached_meta)
    check("cache hit is faster", cached_meta.get("latency_ms", 1 << 30) < live_meta.get("latency_ms", 0), f"{cached_meta.get('latency_ms')} vs {live_meta.get('latency_ms')}")

    status, memory = chat(first_session, "What did I ask you in my previous message? Answer briefly.")
    check("second turn succeeds", status == 200, memory)
    status, stored = request(base, "GET", f"/sessions/{first_session}/messages")
    roles = [m.get("role") for m in stored.get("messages", [])]
    check("history stored in order", status == 200 and roles == ["user", "assistant", "user", "assistant"], roles)

    for code in (429, 500):
        status, failover = chat(sid(), f"Say hello in one word. Run {uuid.uuid4().hex[:8]}.", f"key_1={code}")
        meta = failover.get("metadata", {})
        check(f"key_1={code} fails over to key_2", status == 200 and meta.get("key_id") == "key_2" and meta.get("fallback_attempts") == 1, failover)

    outage_session = sid()
    status, outage = chat(outage_session, "This must not be stored.", "*=429")
    check("*=429 returns 503", status == 503 and outage == {"detail": "all providers unavailable"}, f"{status} {outage}")
    status, _ = request(base, "GET", f"/sessions/{outage_session}/messages")
    check("failed request not stored", status == 404, status)

    status, body = request(base, "POST", "/chat", {"session_id": sid(), "message": "x" * 4001})
    check("over-long message returns 422", status == 422 and "detail" in body, status)
    status, body = request(base, "POST", "/chat", b"{not json")
    check("malformed JSON returns 422", status == 422 and "detail" in body, status)
    status, body = request(base, "GET", f"/sessions/{sid()}/messages")
    check("unknown session returns 404", status == 404 and body == {"detail": "session not found"}, f"{status} {body}")

    print(f"\n{'All checks passed' if not failures else str(len(failures)) + ' check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
