"""Weekly fleet watchdog (PAPER.md §12.6) — did every book actually complete Saturday's cycle?

Runs Sunday 12:00 UTC. For each book: the journal must carry LIVE ``targets`` for the most
recent Friday AND a ``finalized`` record for the same asof. Exits nonzero on any miss, so the
unit's OnFailure= fires the fleet alert. Read-only; no lock needed (journals are append-only
and a torn final line is tolerated by Journal.read)."""

from __future__ import annotations

import datetime as dt
import sys

sys.path.insert(0, "src")

from convexity.paper.books import BOOKS  # noqa: E402
from convexity.paper.journal import Journal  # noqa: E402


def last_friday_iso(today: dt.date | None = None) -> str:
    d = today or dt.datetime.now(dt.timezone.utc).date()
    return str(d - dt.timedelta(days=(d.weekday() - 4) % 7))


def check(today: dt.date | None = None) -> list[str]:
    friday = last_friday_iso(today)
    problems: list[str] = []
    for spec in BOOKS.values():
        j = Journal(spec.journal_path)
        t = j.last("targets")
        if t is None or t.get("asof") != friday or t.get("mode") != "live":
            problems.append(f"{spec.name}: no LIVE targets for {friday} "
                            f"(last: {t.get('asof') if t else None}/{t.get('mode') if t else None})")
        f = j.last("finalized")
        if f is None or f.get("asof") != friday:
            problems.append(f"{spec.name}: no finalized record for {friday} "
                            f"(last: {f.get('asof') if f else None})")
        k = j.last("killed")
        if k is not None and t is not None and int(k.get("ts", 0)) > int(t.get("ts", 0)):
            # a killed book intentionally skips sessions — downgrade to informational
            problems = [p for p in problems if not p.startswith(spec.name)]
            print(f"[watchdog] {spec.name}: killed after last targets — skipping checks "
                  f"(operator halt is not a failure)")
    return problems


def main() -> None:
    problems = check()
    friday = last_friday_iso()
    if problems:
        for p in problems:
            print(f"[watchdog] MISS: {p}")
        raise SystemExit(f"[watchdog] {len(problems)} problem(s) for week {friday} — "
                         f"alert fires via OnFailure")
    print(f"[watchdog] all books completed the {friday} cycle (targets+finalized present)")


if __name__ == "__main__":
    main()
