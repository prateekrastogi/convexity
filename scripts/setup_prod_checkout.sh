#!/bin/bash
# Prod deployment bootstrap (PAPER.md §12.6): SEPARATE checkout so the demo fleet keeps running
# as the shadow book. Idempotent — safe to re-run after git pulls.
#
#   /home/speculation/convexity       = demo fleet (shadow), owns the lake + data maintenance
#   /home/speculation/convexity-prod  = prod fleet: own journals/lock/.env/venv; lake via
#                                       read-only symlinks; prod-named systemd units installed
#                                       DISABLED and staggered +5 min after demo
set -euo pipefail
SRC=/home/speculation/convexity
DST=/home/speculation/convexity-prod

# 1) checkout (clone from the local repo for speed; origin repointed to GitHub for pulls)
if [ ! -d "$DST/.git" ]; then
  git clone -q "$SRC" "$DST"
  git -C "$DST" remote set-url origin git@convexity.github.com:prateekrastogi/convexity.git
fi
git -C "$DST" fetch -q origin && git -C "$DST" merge -q --ff-only origin/main

# 2) data: shared read-only lake via symlinks; OWN paper state (journals, ledgers, lock)
mkdir -p "$DST/data/paper"
for sub in cot curves dukascopy earnings funding klines stitched universe; do
  [ -e "$DST/data/$sub" ] || ln -s "$SRC/data/$sub" "$DST/data/$sub"
done

# 3) .env template — the operator fills the prod keys (NEVER stored anywhere else)
if [ ! -f "$DST/.env" ]; then
  cat > "$DST/.env" <<'EOF'
CONVEXITY_VENUE=prod
# REQUIRED before cutover — Binance USDT-M futures PROD keys (enable futures; IP-restrict to
# this VM; withdrawal permission OFF):
BINANCE_FUTURES_API_KEY=
BINANCE_FUTURES_API_SECRET=
# optional alert push (also honored by the demo fleet's .env)
#NTFY_TOPIC=
#TELEGRAM_BOT_TOKEN=
#TELEGRAM_CHAT_ID=
EOF
  chmod 600 "$DST/.env"
fi

# 4) own venv
cd "$DST" && ~/.local/bin/uv sync -q

# 5) prod systemd units: demo units transformed (paths -> prod checkout, +5 min stagger,
#    " (PROD)" descriptions), installed DISABLED. Data-maintenance units (lake-refresh,
#    datamaint, discovery) are NOT duplicated — the demo checkout is the single lake writer.
UNITS="paper-rebalance paper-finalize paper-recon commodity-rebalance commodity-finalize \
commodity-recon equity-rebalance equity-finalize equity-recon watchdog backup"
shift_cal() {  # +5 min on the OnCalendar minute field
  case "$1" in
    *"00:10:00"*) echo "${1/00:10:00/00:15:00}";;
    *"00:25:00"*) echo "${1/00:25:00/00:30:00}";;
    *"00:40:00"*) echo "${1/00:40:00/00:45:00}";;
    *"00:45:00"*) echo "${1/00:45:00/00:50:00}";;
    *"01:05:00"*) echo "${1/01:05:00/01:10:00}";;
    *"01:10:00"*) echo "${1/01:10:00/01:15:00}";;
    *"02:20:00"*) echo "${1/02:20:00/02:25:00}";;
    *"02:50:00"*) echo "${1/02:50:00/02:55:00}";;
    *"03:20:00"*) echo "${1/03:20:00/03:25:00}";;
    *"06:10:00"*) echo "${1/06:10:00/06:20:00}";;
    *"12:00:00"*) echo "${1/12:00:00/12:10:00}";;
    *) echo "$1";;
  esac
}
TMPD=$(mktemp -d)
for u in $UNITS; do
  for ext in service timer; do
    src_unit="$SRC/deploy/convexity-$u.$ext"
    [ -f "$src_unit" ] || continue
    dst_unit="$TMPD/convexity-prod-$u.$ext"
    if [ "$ext" = timer ]; then
      while IFS= read -r line; do
        case "$line" in
          OnCalendar=*) shift_cal "$line";;
          Description=*) echo "$line (PROD)";;
          *) echo "$line";;
        esac
      done < "$src_unit" > "$dst_unit"
    else
      sed -e "s|/home/speculation/convexity/|/home/speculation/convexity-prod/|g" \
          -e "s|^Description=\(.*\)$|Description=\1 (PROD)|" "$src_unit" > "$dst_unit"
    fi
  done
done
# backup: pass the prod repo as arg (parameterized journal_backup.sh)
sed -i "s|journal_backup.sh$|journal_backup.sh /home/speculation/convexity-prod|" \
    "$TMPD/convexity-prod-backup.service"
sudo cp "$TMPD"/convexity-prod-*.service "$TMPD"/convexity-prod-*.timer /etc/systemd/system/
rm -rf "$TMPD"
sudo systemctl daemon-reload
echo "[setup-prod] units installed (DISABLED — the cutover checklist enables them):"
ls /etc/systemd/system/convexity-prod-*.timer | xargs -n1 basename
echo "[setup-prod] next: fill $DST/.env keys, then run the preflight:"
echo "  cd $DST && set -a && source .env && set +a && .venv/bin/python scripts/prod_preflight.py"
