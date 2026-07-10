# deploy/ — paper-rebalancer systemd units (PAPER.md §6-§8)

Source of truth for the VM's scheduled sessions. Install (copies, not symlinks — updating the
deployed units is an explicit step, so deploys stay intentional):

```bash
sudo cp deploy/convexity-*.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now convexity-paper-rebalance.timer convexity-paper-finalize.timer \
                            convexity-paper-recon.timer convexity-lake-refresh.timer
```

Weekly cadence (all UTC): Sat 00:10 rebalance (Friday close +10min, settles the prior week
first) -> Sat 02:20 finalize (cross unfilled maker remainders) -> Sat 04:00 lake refresh; daily
00:25 reconciliation. Every session is a oneshot full pure-recompute — no daemon state to
corrupt; a crashed session is simply re-run (`Persistent=true` catches missed timers after
downtime).

## Runbook
- **First --live session is SUPERVISED** (PAPER.md §8): run `PYTHONPATH=. .venv/bin/python -m
  convexity.paper rebalance --live` manually and watch it before enabling the timers. The
  gateway is EXPERIMENTAL until this session passes.
- **Account setup (once, on the demo account)**: keep the futures account in SINGLE-ASSET margin
  mode so the exchange cannot draft BTC as cross-collateral — the collateral mandate is USDT
  first, USDC only when USDT is exhausted, **BTC never** (PAPER.md §3.1).
- **Deposits/withdrawals**: journal them the same day with
  `python -m convexity.paper flow --amount <signed USDT>` so the TWR chain stays exact
  (OPS.md §15); a ≥10% withdrawal prints an INTERIM_RESIZE directive — execute it.
- **HALT**: the kill-switch never auto-re-arms (OPS.md §6). After investigating, re-arm by
  explicit operator action only.
- **Status / logs**: `python -m convexity.paper status`; `journalctl -u convexity-paper-rebalance
  -n 100`. Dry-run anything with the same commands minus `--live` (no keys needed).

## Commodity book (PAPER.md §11) — separate units, separate monitoring

**STATUS: LIVE on the demo since 2026-07-06** (allocations crypto 8000 / commodity 2000; the
TradFi-Perps agreement was signed via `gateway.sign_tradfi_agreement()` — any new account hits
`-4411` on every TradFi order until this one-time signed call; the demo venue lists only the
metal perps, so the CL-NATGAS pair is venue-gated out until Binance adds them, PAPER.md §11.3).

The commodity RV pair book runs beside the crypto book on the SAME demo account but with its own
journal, symbol domain, targets, timers and logs. Its units are installed by the same `cp` above
and stagger the crypto book (Sat 00:40 rebalance, 02:50 finalize, daily 00:45 recon, Fri 12:00
discovery). For a fresh account, the runbook below MUST pass before enabling:

```bash
sudo systemctl enable --now convexity-commodity-rebalance.timer \
     convexity-commodity-finalize.timer convexity-commodity-recon.timer \
     convexity-commodity-discovery.timer
```

Runbook BEFORE enabling (PAPER.md §11.5):
1. **Allocate both books** — the shared account's equity belongs to two books now:
   `python -m convexity.paper allocate --book crypto --amount <M>` and
   `python -m convexity.paper allocate --book commodity --amount <N>`, N+M ≤ account USDT+USDC.
   The commodity book REFUSES to trade live without an allocation.
2. **One supervised session**: `python -m convexity.paper rebalance --live --book commodity` —
   watch the 5 pairs' legs fill; confirm the orphan sweep logs touch ONLY commodity symbols.
3. Enable the four timers.

Monitor the commodity book independently: `python -m convexity.paper status --book commodity`;
`journalctl -u convexity-commodity-rebalance -n 100`; journal at
`data/paper/commodity/journal.jsonl`, discovery log at `data/paper/commodity/discovery.jsonl`.

## Equity book (PAPER.md §12) — separate units, separate monitoring

**STATUS: MERGED 2026-07-10 (PR #3), TIMERS NOT ENABLED — the §12.4 runbook below must pass
first.** Timed-beta book (owner decision; the certified research verdict on this universe was
negative — EDGE.md §(g)); parity-pinned to the validated D1 construction (Invariant
EQUITY-PARITY, tests/paper/test_equity_parity.py); 300/1000 allocation.

Units (installed by the same `cp` above; staggered after crypto 00:10 and commodity 00:40):
Sat 01:10 rebalance, Sat 03:20 finalize, daily 01:05 recon, Sat 05:30 data maintenance
(dukascopy new-symbol backfill + split watch — PAPER.md §12.3).

Runbook BEFORE enabling (PAPER.md §12.4 — supervised, touches live account state):
1. Verify position-margin mode is CROSSED for traded symbols; asset mode stays SINGLE-ASSET
   (PAPER.md §3.1 — the BTC mandate; two different axes).
2. `python -m convexity.paper allocate-split --total <ACCOUNT_USDT>` — re-allocates ALL THREE
   books 400/300/300 under the account lock.
3. One supervised session: `python -m convexity.paper rebalance --live --book equity`, then
   `finalize --live --book equity` ~2h later — watch the 3x leverage dial, maker fills, margin/
   liq-distance journal records, and that the orphan sweep touches ONLY equity-domain symbols.
4. `sudo systemctl enable --now convexity-equity-rebalance.timer convexity-equity-finalize.timer
   convexity-equity-recon.timer convexity-equity-datamaint.timer`

Kill switch (any book, any time): `python -m convexity.paper kill --book equity [--now]`;
resume with `--resume`. Monitor: `status --book equity`; `journalctl -u
convexity-equity-rebalance -n 100`; journal at `data/paper/equity/journal.jsonl`; split-watch
flags at `data/universe/equity_flags.json` (clear a flag by setting `"cleared": true`).

## Manual positions (operator carve-out, PAPER.md §12.5)

Trade freely from the Binance app — attribution is automatic. Every gateway order is
stamped `cvx-…`; any other fill is adopted into `data/paper/manual_reserve.json` at the next
sync (every recon/rebalance/finalize/kill, inside the account lock). Books net the reserve out
of sizing, orphan sweeps, kills and finalize; manual closes auto-release.

```bash
python -m convexity.paper manual               # read-only: reserves + watermark age
python -m convexity.paper manual --bootstrap   # PROD CUTOVER, once, BEFORE enabling timers:
                                               # seeds ledger with all existing venue positions
                                               # (refuses unless the account is virgin)
```

Wrong adoption? Hand-edit the ledger JSON. A reserve exceeding the venue position clamps the
book share to 0 and journals the discrepancy (never inverts).

## Production hardening (PAPER.md §12.6)

- `CONVEXITY_VENUE=demo|prod` in `.env` is the ONLY venue switch (prod also needs
  `BINANCE_FUTURES_API_KEY/_SECRET`). Every live session journals its venue resolution.
- Alerts: unit failures append to `data/paper/ALERTS.jsonl` and push via `NTFY_TOPIC` or
  `TELEGRAM_BOT_TOKEN`+`TELEGRAM_CHAT_ID` if set in `.env`. Rebalance/finalize auto-retry once
  after 15 min. Watchdog Sun 12:00 UTC verifies the Saturday cycle completed.
- Backups: nightly 06:10 UTC to `/home/speculation/backups` + the repo's `journal-backups`
  branch. Restore = untar over `data/paper/`.

## Prod deployment (separate checkout, PAPER.md §12.6)

The prod fleet runs from `/home/speculation/convexity-prod` — its own journals, account lock,
`.env` (CONVEXITY_VENUE=prod + `BINANCE_FUTURES_API_KEY/_SECRET`) and venv; the lake is shared
read-only via symlinks. The demo checkout keeps running untouched as the shadow book and stays
the ONLY lake writer (lake-refresh/datamaint/discovery are not duplicated).

```bash
bash scripts/setup_prod_checkout.sh        # idempotent: checkout, symlinks, .env template,
                                           # venv, prod units installed DISABLED (+5min stagger)
# fill /home/speculation/convexity-prod/.env with the prod keys, then:
cd /home/speculation/convexity-prod
set -a; source .env; set +a
.venv/bin/python scripts/prod_preflight.py # read-only checklist; exit 0 = ready
```

Cutover (first cycle SUPERVISED, §12.6): preflight green -> `manual --bootstrap` (if you hold
positions) -> `allocate-split --total <prod USDT>` -> run rebalance/finalize by hand watching
the venue record and `finalized mode=ids` -> `sudo systemctl enable --now
convexity-prod-{paper,commodity,equity}-{rebalance,finalize,recon}.timer
convexity-prod-watchdog.timer convexity-prod-backup.timer`. Backups land in
`/home/speculation/backups/<checkout>` and per-checkout `journal-backups-<checkout>` branches.
