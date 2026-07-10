#!/bin/bash
# Fleet failure alert (PAPER.md §12.6): durable local record + best-effort push.
# Invoked by systemd OnFailure=convexity-alert@%n.service with the failed unit as $1.
# Never exits nonzero (an alert failure must not cascade).
UNIT="${1:-unknown}"
TS=$(date -u +%FT%TZ)
cd /home/speculation/convexity 2>/dev/null || exit 0
mkdir -p data/paper
printf '{"ts":"%s","unit":"%s","host":"%s"}\n' "$TS" "$UNIT" "$(hostname)" >> data/paper/ALERTS.jsonl
set -a; source .env 2>/dev/null; set +a
MSG="convexity FLEET ALERT: $UNIT failed at $TS on $(hostname) — journalctl -u $UNIT"
if [ -n "$NTFY_TOPIC" ]; then
  curl -fsS -m 10 -d "$MSG" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
fi
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
  curl -fsS -m 10 "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
    -d chat_id="$TELEGRAM_CHAT_ID" -d text="$MSG" >/dev/null 2>&1 || true
fi
exit 0
