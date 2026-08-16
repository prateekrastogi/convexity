#!/usr/bin/env python3
"""Restore systemd's ``Persistent=true`` on top of cron.

systemd ran a missed timer once on next boot. cron has no catch-up at all, so the Docker
migration silently dropped that behaviour: a job due while the container was down was skipped
outright and nothing recorded it. On a fleet whose jobs are the weekly rebalance and the daily
reconciliation, that is a correctness gap, not an inconvenience.

At container start this reads the same crontab supercronic will run, and for each job compares
the most recent scheduled fire time against the last successful run recorded in
``data/ops/lastrun.json`` (bind-mounted, so it survives restarts and image rebuilds). A job whose
last scheduled fire was missed is run ONCE, immediately, sequentially — matching systemd, which
triggers the unit a single time regardless of how many fires were missed.

TWO DELIBERATE DEVIATIONS FROM systemd, both toward safety:

  1. MAX AGE (default 48h, ``CONVEXITY_CATCHUP_MAX_AGE_H``). systemd would run a missed job no
     matter how stale. Here a fire older than the cap is skipped and reported. Rationale: a
     Saturday rebalance recovered on Sunday is right; the same rebalance recovered on Wednesday
     would re-risk the book mid-week off a stale schedule, and that is an operator decision, not
     an automatic one. 48h covers a weekend-long outage.

  2. FIRST BOOT SEEDS, NEVER STAMPEDES. With no prior state every one of the 34 jobs would look
     missed and fire at once — including six rebalances across two venues. First run records
     "now" for every job and executes nothing.

Read-only with respect to the venue: it does not decide anything, it only re-runs jobs the
schedule already called for.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

from croniter import croniter

CRONTAB = pathlib.Path(os.environ.get("CONVEXITY_CRONTAB", "/app/ops/docker/crontab"))
# ONE SMALL FILE PER JOB, not a shared JSON document: supercronic runs jobs concurrently, and
# several bash processes rewriting the same JSON would interleave and corrupt it. Per-job files
# make the write trivially safe and the state trivially inspectable.
STATE_DIR = pathlib.Path(os.environ.get("CONVEXITY_LASTRUN_DIR", "/app/data/ops/lastrun"))
MAX_AGE_H = float(os.environ.get("CONVEXITY_CATCHUP_MAX_AGE_H", "48"))


def parse_crontab(text: str) -> list[tuple[str, str, str]]:
    """-> [(job_name, cron_expr, shell_command)]. Mirrors how supercronic reads the same file."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" in line.split()[0]:
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        expr, command = " ".join(parts[:5]), parts[5]
        # job name is the argument right after job.sh (possibly behind a VAR=... prefix)
        name = None
        toks = command.split()
        for i, t in enumerate(toks):
            if t.endswith("job.sh") and i + 1 < len(toks):
                name = toks[i + 1]
                break
        if name is None:
            continue
        out.append((name, expr, command))
    return out


def read_last(name: str) -> datetime | None:
    try:
        return datetime.fromisoformat(( STATE_DIR / name).read_text().strip())
    except Exception:
        return None


def write_last(name: str, when: datetime) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_DIR / f".{name}.tmp"
    tmp.write_text(when.isoformat())
    tmp.replace(STATE_DIR / name)


def main() -> int:
    now = datetime.now(timezone.utc)
    if not CRONTAB.exists():
        print(f"[catchup] no crontab at {CRONTAB}, nothing to do")
        return 0

    jobs = parse_crontab(CRONTAB.read_text())
    first_boot = not STATE_DIR.exists() or not any(STATE_DIR.iterdir())

    if first_boot:
        for name, _expr, _cmd in jobs:
            write_last(name, now)
        print(f"[catchup] first boot — seeded {len(jobs)} jobs at {now.isoformat()}, ran none")
        return 0

    missed, skipped_stale, ran_ok, ran_fail = [], [], [], []
    for name, expr, command in jobs:
        prev_fire = croniter(expr, now).get_prev(datetime)
        last = read_last(name)
        if last is None:
            # A job added to the crontab since the last boot: seed it, do not back-run it.
            write_last(name, now)
            continue
        if last >= prev_fire:
            continue
        age_h = (now - prev_fire).total_seconds() / 3600.0
        if age_h > MAX_AGE_H:
            skipped_stale.append((name, prev_fire, age_h))
            continue
        missed.append((name, prev_fire, command))

    for name, prev_fire, _age in skipped_stale:
        print(f"[catchup] SKIP {name}: missed fire {prev_fire.isoformat()} is older than "
              f"{MAX_AGE_H}h — re-run by hand if you still want it")

    if not missed:
        print(f"[catchup] nothing missed ({len(jobs)} jobs checked)")
        return 0

    print(f"[catchup] {len(missed)} missed job(s) to recover, running sequentially")
    for name, prev_fire, command in missed:
        print(f"[catchup] RUN {name} (missed {prev_fire.isoformat()})")
        rc = subprocess.call(["/bin/bash", "-c", command])
        (ran_ok if rc == 0 else ran_fail).append(name)
        # job.sh stamps its own success. A FAILED job is deliberately left unstamped so the next
        # boot sees it as still-missed and tries again, rather than recording it as done.

    print(f"[catchup] done: {len(ran_ok)} ok, {len(ran_fail)} failed"
          + (f" ({', '.join(ran_fail)})" if ran_fail else ""))
    # Never block startup on a failed catch-up — supercronic must come up regardless.
    return 0


if __name__ == "__main__":
    sys.exit(main())
