#!/bin/bash
# One fleet job, run inside the container by supercronic.
#
# Reproduces the three things systemd provided beyond scheduling:
#   1. OnFailure=convexity-alert@%n  -> immediate durable record + telegram/ntfy push
#   2. convexity-retry@%n            -> one delayed retry 900s after a failure
#   3. journald                      -> stdout/stderr, captured by the docker log driver
#
# The retry is INLINE (sleep in-process) rather than backgrounded: a detached retry would be
# orphaned by a container restart and would leave supercronic with no idea a job was still
# pending. Jobs are idempotent by design (settle guard, venue-truth carried book, resting-order
# cancel), so re-running after a transient failure is safe — the same property systemd's
# retry@ unit relied on.
#
# Usage: job.sh <job-name> <command...>
set -uo pipefail

NAME="${1:?job name required}"
shift

RETRY_DELAY="${CONVEXITY_RETRY_DELAY:-900}"

alert() {
    # Never fail the job because alerting failed — mirrors fleet_alert.sh's contract.
    local ts msg
    ts=$(date -u +%FT%TZ)
    mkdir -p /app/data/paper 2>/dev/null || true
    printf '{"ts":"%s","unit":"%s","host":"%s"}\n' \
        "$ts" "$NAME" "$(hostname)" >> /app/data/paper/ALERTS.jsonl 2>/dev/null || true
    msg="convexity FLEET ALERT: $NAME failed at $ts on $(hostname) — docker logs convexity-fleet"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -m 10 -d "$msg" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    fi
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -fsS -m 10 "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d chat_id="$TELEGRAM_CHAT_ID" -d text="$msg" >/dev/null 2>&1 || true
    fi
}

stamp_success() {
    # Records the last SUCCESSFUL run so catchup_missed.py can tell, at next container start,
    # whether this job's most recent scheduled fire was missed — the Persistent=true behaviour
    # cron does not have. One file per job: supercronic runs jobs concurrently and a shared
    # document would interleave. Written via tmp+mv so a torn file is never observable.
    # MUST honour the same override catchup_missed.py reads. When these two disagree, job.sh
    # stamps one directory while the catch-up reads another, so every job looks permanently
    # missed and is re-run on every single container start. Caught in testing.
    local dir="${CONVEXITY_LASTRUN_DIR:-/app/data/ops/lastrun}"
    mkdir -p "$dir" 2>/dev/null || return 0
    date -u +%FT%T+00:00 > "$dir/.$NAME.tmp" 2>/dev/null \
        && mv -f "$dir/.$NAME.tmp" "$dir/$NAME" 2>/dev/null || true
}

cd /app || exit 1

# NOTE: capture $? IMMEDIATELY after the command, never after an `if`. A bash `if` whose
# condition fails and which has no `else` returns 0 ITSELF, so `if "$@"; then ...; fi; rc=$?`
# silently reports success for every failure — the wrapper then exits 0 and supercronic never
# learns the job failed. Caught by testing the failure path before cutover.
echo "[job:$NAME] start $(date -u +%FT%TZ) venue=${CONVEXITY_VENUE:-demo}"
"$@"
rc=$?
if [ "$rc" -eq 0 ]; then
    stamp_success
    echo "[job:$NAME] ok $(date -u +%FT%TZ)"
    exit 0
fi

echo "[job:$NAME] FAILED rc=$rc — alerting, retrying in ${RETRY_DELAY}s" >&2
alert
sleep "$RETRY_DELAY"

echo "[job:$NAME] retry $(date -u +%FT%TZ)"
"$@"
rc=$?
if [ "$rc" -eq 0 ]; then
    stamp_success
    echo "[job:$NAME] ok-on-retry $(date -u +%FT%TZ)"
    exit 0
fi

echo "[job:$NAME] FAILED-ON-RETRY rc=$rc" >&2
alert
exit "$rc"
