# Build stage: install hash-locked dependencies into a virtualenv.
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes -r /tmp/requirements.txt \
    && /opt/venv/bin/pip uninstall -y pip setuptools 2>/dev/null || true
# The LiteLLM proxy admin UI is never served by this app.
RUN rm -rf /opt/venv/lib/python3.12/site-packages/litellm/proxy/_experimental/out \
    && find /opt/venv -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +

# Runtime stage: only the virtualenv and application code.
FROM python:3.12-slim
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown app:app /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY app ./app
COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    DATABASE_URL=sqlite:////data/app.db \
    CACHE_DIR=/data/llm-cache
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request as r; r.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=4)"]
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
