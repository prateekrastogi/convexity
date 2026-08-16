# convexity — self-contained runtime for the three-book fleet.
#
# The container carries its own scheduler (supercronic) and runs every one of the 31 fleet jobs
# internally, so the deployment depends on nothing host-side: no systemd, no .venv, no host
# python. `docker compose up -d` on any Docker host is the whole deployment.
#
# supercronic rather than systemd-in-a-container: systemd as PID 1 needs --privileged or careful
# cgroup delegation and turns the container into a pet VM. supercronic is built for this — PID 1,
# logs to stdout, forwards signals, no setuid.
#
# What is deliberately NOT in the image:
#   * data/  — 24 GB of lake + journals, and the fleet's accounting truth. Bind-mounted at
#              runtime. The per-venue flock (data/<venue>/account.lock) only serialises correctly
#              if every process shares one inode, which a bind mount guarantees.
#   * .env   — secrets injected at runtime, never layered.
#   * ~/.ssh — the backup job's deploy key, mounted read-only.
#
# Runs as uid/gid 1001 to match the host `ubuntu` owner of data/, so journals written from inside
# the container stay readable by host tooling. Host is aarch64.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependency layer first: the expensive step, re-runs only when the lock changes.
# README.md rides along because pyproject sets `readme = "README.md"` — hatchling validates it
# when building the project below, and its absence fails the build with a bare OSError.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

# curl  -> the alert wrapper's telegram/ntfy push (fleet_alert.sh)
# git   -> journal_backup.sh force-pushes the nightly state to a private branch
# ssh   -> that push authenticates with the mounted deploy key
# tar   -> the backup tarball itself
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates tzdata curl git openssh-client tar \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1001 app \
    && useradd -u 1001 -g 1001 -m -s /bin/bash app

ARG SUPERCRONIC_VERSION=v0.2.33
ARG SUPERCRONIC_ARCH=arm64
RUN curl -fsSL -o /usr/local/bin/supercronic \
      "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${SUPERCRONIC_ARCH}" \
    && chmod +x /usr/local/bin/supercronic

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY deploy/ ./deploy/
COPY pyproject.toml uv.lock README.md ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    CONVEXITY_HOME=/app

# data/ and backups/ are mount points; create them owned by app so a missing bind mount fails
# loudly rather than silently writing into the image layer.
# journal_backup.sh derives the backup dir and the remote branch name from the repo path's
# basename (NAME=$(basename "$REPO") -> journal-backups-convexity). Pointing it at /app would
# silently rename the branch to journal-backups-app and orphan the existing backup history, so
# the old absolute path is kept as a symlink and the cron entry passes it verbatim.
RUN mkdir -p /app/data /home/speculation \
    && ln -s /app /home/speculation/convexity \
    && chmod +x /app/deploy/docker/*.sh /app/scripts/*.sh 2>/dev/null || true \
    && chown -R 1001:1001 /app \
    && chown -h 1001:1001 /home/speculation/convexity

USER 1001:1001

HEALTHCHECK NONE

# Default: the interpreter, so one-off/manual runs work exactly like the old venv python.
#   docker run --rm convexity:latest -m convexity.paper status
# The long-running fleet container overrides this with supercronic (see docker-compose.yml).
ENTRYPOINT ["/app/.venv/bin/python"]
CMD ["-c", "import convexity, sys; print('convexity image OK', sys.version.split()[0])"]
