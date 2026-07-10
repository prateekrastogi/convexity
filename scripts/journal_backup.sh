#!/bin/bash
# Nightly paper-state backup (PAPER.md §12.6): journals + ledgers are the fleet's accounting
# truth and exist nowhere else. Local rotation (30) + off-VM copy on a dedicated git branch
# (private repo; single force-pushed commit keeps remote size = current backup set).
set -e
cd /home/speculation/convexity
TS=$(date -u +%Y%m%d)
BK=/home/speculation/backups
mkdir -p "$BK"
tar czf "$BK/paperstate-$TS.tar.gz" --ignore-failed-read \
  data/paper/journal.jsonl \
  data/paper/commodity/journal.jsonl \
  data/paper/equity/journal.jsonl \
  data/paper/manual_reserve.json \
  data/paper/manual_sync.json \
  data/paper/ALERTS.jsonl \
  data/universe/equity_flags.json 2>/dev/null || true
ls -1t "$BK"/paperstate-*.tar.gz 2>/dev/null | tail -n +31 | xargs -r rm
# off-VM: single-commit force-push to the journal-backups branch (fresh history each night —
# remote stores only the current rotation set, never grows unbounded)
cd "$BK"
rm -rf .git
git init -q -b journal-backups
git remote add origin git@convexity.github.com:prateekrastogi/convexity.git
git add -A
git -c user.email=fleet@arb-vm -c user.name="convexity fleet" commit -q -m "paper-state backup $TS"
git push -q --force origin journal-backups
echo "[backup] $TS: $(ls "$BK"/paperstate-*.tar.gz | wc -l) local tarballs; branch pushed"
