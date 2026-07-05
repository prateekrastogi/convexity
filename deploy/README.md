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

The commodity RV pair book runs beside the crypto book on the SAME demo account but with its own
journal, symbol domain, targets, timers and logs. Its units are installed by the same `cp` above
but are **enabled separately and ONLY after the runbook below** (they stagger the crypto book:
Sat 00:40 rebalance, 02:50 finalize, daily 00:45 recon, Fri 12:00 discovery):

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
