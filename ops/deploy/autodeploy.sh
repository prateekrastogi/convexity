#!/usr/bin/env bash
# convexity continuous deployment — ONE script, runnable in the background on any VM.
#
#   nohup ops/deploy/autodeploy.sh > /var/log/convexity-autodeploy.log 2>&1 &
#   ops/deploy/autodeploy.sh --once      # single pass, for cron or a manual deploy
#
# Watches the GitHub origin for new commits on the tracked branch and, when one lands:
#   1. fast-forwards the checkout,
#   2. builds the image — the Dockerfile's `test` stage gates this, so a red suite = no image,
#   3. runs the LAKE-DEPENDENT suite (the equity parity gate) against the real mounted data,
#      which the in-image gate cannot do because data/ is 24 GB and not in the build context,
#   4. only then promotes the image and recreates the container.
#
# SAFETY PROPERTIES, in order of how much they matter on a live trading fleet:
#
#   * THE RUNNING FLEET IS NEVER TAKEN DOWN FOR A BUILD. Everything is built and tested under a
#     throwaway tag; the live tag only moves after every gate is green. A failed build leaves the
#     current container running, untouched.
#   * IT WILL NOT RESTART MID-CYCLE. If a fleet job is executing (the Saturday rebalance can hold
#     the account lock for minutes while resting maker orders), deployment defers to the next
#     poll rather than killing a live trading session.
#   * IT NEVER FORCE-RESETS THE CHECKOUT. --ff-only; local divergence stops the deploy loudly
#     instead of silently discarding work.
#   * FAILURES ALERT AND STOP. No partial promotion, no retry storm.
set -uo pipefail

REPO="${CONVEXITY_REPO:-/home/speculation/convexity}"
BRANCH="${CONVEXITY_BRANCH:-main}"
INTERVAL="${CONVEXITY_POLL_INTERVAL:-60}"
IMAGE="${CONVEXITY_IMAGE:-convexity}"
SERVICE="${CONVEXITY_SERVICE:-fleet}"
CONTAINER="${CONVEXITY_CONTAINER:-convexity-fleet}"
LOCK="${CONVEXITY_DEPLOY_LOCK:-/tmp/convexity-autodeploy.lock}"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

log() { echo "[autodeploy $(date -u +%FT%TZ)] $*"; }

alert() {
    # Best-effort; never fail the deploy because the notification failed.
    local msg="convexity DEPLOY: $*"
    log "$msg"
    [ -f "$REPO/.env" ] && { set -a; . "$REPO/.env" 2>/dev/null; set +a; }
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -fsS -m 10 "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d chat_id="$TELEGRAM_CHAT_ID" -d text="$msg" >/dev/null 2>&1 || true
    fi
}

# A fleet job in flight holds the account lock and may have resting orders on the venue.
# Restarting the container underneath it is the one thing this script must never do.
fleet_busy() {
    docker exec "$CONTAINER" bash -c \
        'pgrep -f "convexity.paper" >/dev/null || pgrep -f "job.sh" >/dev/null' 2>/dev/null
}

deploy() {
    local sha_short="$1"
    local tag_new="${IMAGE}:candidate-${sha_short}"

    log "building $tag_new (test suite gates this build)"
    if ! docker build -t "$tag_new" "$REPO" >/tmp/convexity-build.log 2>&1; then
        alert "BUILD FAILED at $sha_short — fleet left running on the previous image. $(tail -3 /tmp/convexity-build.log | tr '\n' ' ')"
        return 1
    fi

    # Gate the in-image suite cannot run: the equity parity test needs the stitched lake, which
    # is a 24 GB bind mount, not build context. It self-skips when absent — so without this step
    # a construction-breaking change would sail through.
    # The FULL suite against the mounted lake — not just the slow marker. This is the only place
    # the equity parity gate can actually run (it self-skips without data/stitched), and the only
    # place total coverage is honestly measurable, so it is where the 90% floor is enforced.
    log "running FULL suite + 90% coverage floor against mounted data"
    if ! docker run --rm --user 1001:1001 \
            -v "$REPO/data:/app/data" -e PYTHONPATH=/app -e COVERAGE_FILE=/tmp/.cov \
            --entrypoint /app/.venv/bin/python "$tag_new" \
            -m pytest -q -p no:cacheprovider \
            --cov=convexity --cov-report=term --cov-fail-under=90 \
            >/tmp/convexity-slow.log 2>&1; then
        alert "FULL SUITE OR COVERAGE FLOOR FAILED at $sha_short — NOT deployed. $(tail -3 /tmp/convexity-slow.log | tr '\n' ' ')"
        docker rmi -f "$tag_new" >/dev/null 2>&1 || true
        return 1
    fi

    if fleet_busy; then
        log "a fleet job is running — deferring promotion to the next poll"
        return 2
    fi

    log "promoting $tag_new -> ${IMAGE}:latest and recreating $CONTAINER"
    docker tag "$tag_new" "${IMAGE}:latest"
    if ! (cd "$REPO" && docker compose up -d --force-recreate "$SERVICE" >>/tmp/convexity-build.log 2>&1); then
        alert "RECREATE FAILED at $sha_short — check docker compose ps immediately"
        return 1
    fi

    sleep 5
    if ! docker ps --filter "name=$CONTAINER" --filter "status=running" -q | grep -q .; then
        alert "CONTAINER NOT RUNNING after deploy of $sha_short — check docker logs $CONTAINER"
        return 1
    fi

    docker rmi -f "$tag_new" >/dev/null 2>&1 || true
    docker image prune -f >/dev/null 2>&1 || true
    alert "deployed $sha_short OK — fleet up"
    return 0
}

pass() {
    cd "$REPO" || { log "repo $REPO missing"; return 1; }

    # WHICH branch is checked out matters as much as which SHA. A feature branch that already
    # contains origin/$BRANCH would make the --ff-only merge below a no-op, and the script would
    # build and promote THAT branch while reporting it had deployed $BRANCH.
    local current
    current=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
    if [ "$current" != "$BRANCH" ]; then
        log "checkout is on '$current', not '$BRANCH' — refusing to deploy (set CONVEXITY_BRANCH to change target)"
        return 1
    fi

    git fetch -q origin "$BRANCH" 2>/dev/null || { log "fetch failed (network?), will retry"; return 0; }
    local local_sha remote_sha
    local_sha=$(git rev-parse HEAD)
    remote_sha=$(git rev-parse "origin/$BRANCH")
    [ "$local_sha" = "$remote_sha" ] && return 0

    log "new commit on origin/$BRANCH: ${remote_sha:0:8} (have ${local_sha:0:8})"

    # --ff-only on purpose: if the checkout has diverged, stop and say so rather than throw work
    # away. A deploy script must never be the thing that loses a commit.
    if ! git merge --ff-only "origin/$BRANCH" >/dev/null 2>&1; then
        alert "checkout has DIVERGED from origin/$BRANCH — refusing to auto-deploy. Resolve by hand."
        return 1
    fi

    deploy "${remote_sha:0:8}"
}

# One deployer at a time, even if the script is started twice.
exec 9>"$LOCK"
if ! flock -n 9; then
    log "another autodeploy holds $LOCK — exiting"
    exit 0
fi

log "watching origin/$BRANCH in $REPO every ${INTERVAL}s (once=$ONCE)"
if [ "$ONCE" = "1" ]; then
    pass
    exit $?
fi

trap 'log "received TERM, exiting"; exit 0' TERM INT
while true; do
    pass
    sleep "$INTERVAL"
done
