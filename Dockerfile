# convexity — self-contained runtime for the three-book fleet (crypto / commodity / equity).
#
# The container carries its own scheduler (supercronic) and runs every fleet job internally, so
# the deployment depends on nothing host-side: no systemd, no .venv, no host python.
# `docker compose up -d` on any Docker host is the whole deployment.
#
# THE TEST SUITE GATES THE IMAGE. The `test` stage runs pytest, and `runtime` copies its pass
# marker — a docker COPY --from a stage forces that stage to build, so a red suite fails the
# build and no image is produced. There is no way to ship an untested image by accident.
#
# Stages: builder (deps) -> test (suite) -> runtime (lean, no dev deps, no tests).
# Only the marker crosses from test into runtime, so gating costs the runtime image nothing.
#
# What is deliberately NOT in the image:
#   * data/  — 24 GB of lake + journals, the fleet's accounting truth. Bind-mounted at runtime.
#              The per-venue flock (data/<venue>/account.lock) only serialises correctly if every
#              process shares one inode, which a bind mount guarantees.
#   * .env   — secrets injected at runtime, never layered.
#   * ~/.ssh — the backup job's deploy key, mounted read-only.
#
# Runs as uid/gid 1001 to match the host `ubuntu` owner of data/. Host is aarch64.

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

# ---------------------------------------------------------------- test (gates the build)
FROM builder AS test

# Dev group (pytest, pytest-cov, ruff) lands only in this stage — runtime never sees it.
RUN uv sync --frozen

COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY ops/ ./ops/

# PATH must include the venv here: the builder sets UV_PROJECT_ENVIRONMENT but never puts
# /app/.venv/bin on PATH, so bare `ruff`/`python` resolve to the system interpreter (or nothing)
# and the gate exits 127 instead of running.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# `not slow` deselects the equity parity gate, which needs the 24 GB stitched lake that is
# deliberately absent from the build context. It would SILENTLY SKIP here (the test self-skips
# when data/stitched is missing), so gating on it in-image would be a lie. The deploy script runs
# the slow suite separately against the mounted data before promoting an image — see
# ops/deploy/autodeploy.sh.
# TESTS gate the build, lint does not. Lint is a style signal; blocking a deploy on it would
# stop a correctness fix from shipping because an import was unsorted. Run `ruff check` in
# development instead — the repo currently has 35 pre-existing findings, tracked in the audit,
# none of them behavioural.
# Regression floor for the IN-IMAGE suite, which runs `-m "not slow"` (the parity gate needs the
# 24 GB lake and is not in the build context) and measures ~88%. Set below that so routine work
# does not trip on noise while a real regression still fails the build. The honest 90% is enforced
# on the FULL suite in ops/deploy/autodeploy.sh, where the lake is mounted.
ARG COV_MIN=85
RUN python -m pytest -q -m "not slow" \
        --cov=convexity --cov-report=term-missing:skip-covered \
        --cov-fail-under=${COV_MIN} \
    && touch /app/.tests-passed

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

# curl -> the alert wrapper's telegram/ntfy push
# git  -> journal_backup.sh force-pushes nightly state to a private branch
# ssh  -> that push authenticates with the mounted deploy key
# tar  -> the backup tarball itself
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

# THE GATE: this COPY forces the `test` stage to build. A failing suite means no marker, which
# means no runtime image. Keep it before the rest so a red suite fails fast.
COPY --from=test /app/.tests-passed /app/.tests-passed

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY ops/ ./ops/
COPY pyproject.toml uv.lock README.md ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# journal_backup.sh derives its backup dir and remote branch from basename($REPO)
# (-> journal-backups-convexity). Pointing it at /app would silently rename the branch to
# journal-backups-app and orphan the existing backup history, so the old absolute path survives
# as a symlink and the crontab passes it verbatim.
RUN mkdir -p /app/data /home/speculation \
    && ln -s /app /home/speculation/convexity \
    && chmod +x /app/ops/docker/*.sh /app/scripts/*.sh 2>/dev/null || true \
    && chown -R 1001:1001 /app \
    && chown -h 1001:1001 /home/speculation/convexity

USER 1001:1001

HEALTHCHECK NONE

# Default: the interpreter, so one-off/manual runs work like the old venv python:
#   docker compose run --rm --entrypoint /app/.venv/bin/python fleet -m convexity.paper status
# The long-running fleet container overrides this with ops/docker/entrypoint.sh, which runs
# the missed-job catch-up and then execs supercronic (see docker-compose.yml).
ENTRYPOINT ["/app/.venv/bin/python"]
CMD ["-c", "import convexity, sys; print('convexity image OK', sys.version.split()[0])"]
