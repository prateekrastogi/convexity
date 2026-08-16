#!/bin/bash
# Fleet container entrypoint: recover anything the schedule missed while we were down, then hand
# PID 1 to the scheduler.
#
# The catch-up is what restores systemd's Persistent=true, which plain cron does not have. It is
# deliberately non-fatal: if recovery fails, supercronic must still come up, because a scheduler
# that refuses to start is a far worse failure than one missed job.
set -uo pipefail

echo "[entrypoint] $(date -u +%FT%TZ) starting fleet container"

if [ "${CONVEXITY_CATCHUP:-1}" = "1" ]; then
    /app/.venv/bin/python /app/ops/docker/catchup_missed.py || \
        echo "[entrypoint] catch-up failed (non-fatal), continuing to scheduler" >&2
else
    echo "[entrypoint] catch-up disabled by CONVEXITY_CATCHUP=0"
fi

echo "[entrypoint] handing off to supercronic"
exec /usr/local/bin/supercronic -passthrough-logs "${CONVEXITY_CRONTAB:-/app/ops/docker/crontab}"
