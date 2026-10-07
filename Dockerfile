# syntax=docker/dockerfile:1.7
#
# One image for every AI-Ready service. The service is chosen at run time by the command
# (`uvicorn aiready.<pkg>.api:create_app --factory ...` or `python -m aiready.ingestion.worker`),
# so a release is a single signed, scanned artefact - the same digest runs in pilot, staging and production.
#
# Stage 1 (builder) compiles wheels with build tools; stage 2 (runtime) carries only the virtualenv,
# runs as a fixed non-root uid and works with a read-only root filesystem (writable paths are volumes).

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /src
# Dependency layer first so code-only changes reuse the cached dependency install.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip wheel \
 && pip install ".[all]" \
 && python -c "import aiready"

# --------------------------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG VERSION=0.0.0-dev
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="aiready" \
      org.opencontainers.image.description="AI-Ready platform services (gateway, guardrails, governance, retrieval, agents, MCP, ingestion, observability)" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://example.invalid/aiready" \
      org.opencontainers.image.licenses="Proprietary" \
      io.aiready.image.kind="platform"

ARG PG_CLIENT_MAJOR=16

# Runtime libraries only: libpq for psycopg; curl and a pg_dump matching the server major version
# (PGDG repo, since Debian's client is older than the pgvector/pg16 server) for the backup CronJobs;
# tini as PID 1 so signals reach uvicorn and zombies are reaped.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg tini \
 && install -d /usr/share/postgresql-common/pgdg \
 && curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc \
 && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends libpq5 "postgresql-client-${PG_CLIENT_MAJOR}" \
 && apt-get purge -y gnupg && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 aiready \
 && useradd --system --uid 10001 --gid 10001 --home-dir /home/aiready --create-home --shell /usr/sbin/nologin aiready \
 && mkdir -p /etc/aiready/config /var/lib/aiready /run/secrets /opt/aiready \
 && chown -R 10001:10001 /var/lib/aiready /home/aiready

COPY --from=builder /opt/venv /opt/venv
# Default configuration baked in; deployments mount their own directory over it.
COPY --chown=10001:10001 config /etc/aiready/config
COPY --chown=10001:10001 deploy/onprem/backup /opt/aiready/backup

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIREADY_CONFIG_DIR=/etc/aiready/config \
    AIREADY_DATA_DIR=/var/lib/aiready \
    AIREADY_JSON_LOGS=1 \
    AIREADY_PORT=8080 \
    HOME=/home/aiready

USER 10001:10001
WORKDIR /home/aiready
VOLUME ["/var/lib/aiready", "/tmp"]
EXPOSE 8080

# Every HTTP service exposes /healthz on its own port; AIREADY_PORT is set per service.
# The ingestion worker overrides this check (it has no HTTP port) in compose/Helm.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('AIREADY_PORT','8080'), timeout=2); sys.exit(0)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "aiready.gateway.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
