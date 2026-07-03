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
