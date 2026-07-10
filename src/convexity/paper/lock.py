"""Account-level advisory lock (PAPER.md §12.2) — ONE lock for the ONE account.

Three books share one exchange account, one API budget, and one allocate/topup flow. Recon
(D1v2) found NO serialization existed — the books relied purely on timer staggering. This module
is the primary mechanism; staggering remains defense-in-depth.

Design:
  * ``fcntl.flock`` on a well-known path in the SHARED lake (``data/paper/account.lock`` resolves
    through every worktree's data symlink to the same inode) — every book/process/worktree
    contends on the same kernel lock.
  * The kernel releases flock automatically when the holder dies (including kill -9), so a
    crashed holder can NEVER wedge the fleet — there is no stale-lock takeover logic to get
    wrong (the classic false-stale/two-owners bug class is absent by construction).
  * A live-but-slow holder blocks waiters up to ``timeout``; on timeout the waiter aborts LOUDLY
    (raises), never proceeds unserialized. Metadata (pid/book/ts) is written into the file for
    operator forensics only — it is never used to "break" the lock.
  * Critical section: everything venue-facing in a rebalance/finalize/recon (account reads →
    margin checks → submissions → orphan sweep) plus allocate/flow journal writes. Signal
    computation stays OUTSIDE.
"""

from __future__ import annotations

import fcntl
import json
import os
import pathlib
import time
from contextlib import contextmanager

from convexity.paper.fleetconfig import ACCOUNT_LOCK_TIMEOUT_S

_POLL_S = 0.5


class AccountLockTimeout(RuntimeError):
    """Another book held the account lock past the timeout — this session aborts rather than
    running unserialized against the shared account. Re-run after the holder finishes."""


@contextmanager
def account_lock(book: str, timeout: float = ACCOUNT_LOCK_TIMEOUT_S,
                 path: str | os.PathLike | None = None):
    if path is None:
        # per-venue lock (PAPER.md §12.6): demo and prod are DIFFERENT exchange accounts —
        # serializing them against each other would be pure false contention
        from convexity.paper.venue import state_root
        path = state_root() / "account.lock"
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a+")
    t0 = time.monotonic()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                waited = time.monotonic() - t0
                if waited >= timeout:
                    holder = "?"
                    try:
                        f.seek(0)
                        holder = f.read(200).strip() or "?"
                    except Exception:
                        pass
                    raise AccountLockTimeout(
                        f"[lock:{book}] account lock not acquired after {waited:.0f}s "
                        f"(holder: {holder}) — aborting rather than running unserialized")
                time.sleep(_POLL_S)
        waited = time.monotonic() - t0
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"pid": os.getpid(), "book": book,
                            "acquired_ts": int(time.time() * 1000)}))
        f.flush()
        print(f"[lock:{book}] account lock acquired after {waited:.1f}s (pid {os.getpid()})")
        yield
    finally:
        if acquired:
            try:
                f.seek(0)
                f.truncate()
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                print(f"[lock:{book}] account lock released")
            except Exception:
                pass  # process teardown releases the flock regardless (kernel guarantee)
        f.close()
