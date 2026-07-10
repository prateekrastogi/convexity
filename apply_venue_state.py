"""Venue-namespaced paper state (owner decision 2026-07-10 night): ONE checkout, BOTH venues.

demo state stays at data/paper/** (journals/history byte-untouched); prod isolates under
data/prod/<book>/journal.jsonl + its own account.lock / manual_reserve.json / manual_sync.json.
Venue selection: code default 'demo'; prod systemd units pin CONVEXITY_VENUE=prod on the
ExecStart line (EnvironmentFile= would override Environment=, so .env must NOT carry the venue).
"""
import pathlib
import sys

repo = pathlib.Path(".")
applied = []


def patch(path: str, old: str, new: str, tag: str) -> None:
    p = repo / path
    t = p.read_text()
    if old not in t:
        sys.exit(f"ANCHOR NOT FOUND [{tag}] in {path}")
    assert t.count(old) == 1, f"anchor ambiguous [{tag}]"
    p.write_text(t.replace(old, new))
    applied.append(tag)


# ---- V1: venue.state_root() -------------------------------------------------------------------
patch("src/convexity/paper/venue.py",
    '''def describe() -> dict:''',
    '''def state_root():
    """Venue-namespaced paper-state root (PAPER.md §12.6): demo keeps the original data/paper
    (journals, lock, ledgers — history byte-untouched); prod isolates under data/prod so BOTH
    fleets run from ONE checkout without ever blending accounting. Lake/universe stay shared
    (venue-agnostic data)."""
    import pathlib

    return pathlib.Path("data/paper" if venue() == "demo" else "data/prod")


def describe() -> dict:''',
    "V1 state_root")

# ---- V2: books.book_journal_path ---------------------------------------------------------------
patch("src/convexity/paper/books.py",
    '''def book_domain(spec: BookSpec) -> frozenset[str]:''',
    '''def book_journal_path(spec: BookSpec) -> str:
    """Venue-resolved journal path (PAPER.md §12.6). Demo keeps the committed legacy paths
    (crypto's original data/paper/journal.jsonl included); prod gets the uniform
    data/prod/<book>/journal.jsonl — a fresh accounting chain per venue by construction."""
    from convexity.paper.venue import state_root, venue

    if venue() == "demo":
        return spec.journal_path
    return str(state_root() / spec.name / "journal.jsonl")


def book_domain(spec: BookSpec) -> frozenset[str]:''',
    "V2 book_journal_path")

# ---- V3: lock resolves per venue ---------------------------------------------------------------
patch("src/convexity/paper/lock.py",
    '''from convexity.paper.fleetconfig import ACCOUNT_LOCK_PATH, ACCOUNT_LOCK_TIMEOUT_S''',
    '''from convexity.paper.fleetconfig import ACCOUNT_LOCK_TIMEOUT_S''',
    "V3a lock import")

patch("src/convexity/paper/lock.py",
    '''@contextmanager
def account_lock(book: str, timeout: float = ACCOUNT_LOCK_TIMEOUT_S,
                 path: str | os.PathLike = ACCOUNT_LOCK_PATH):
    p = pathlib.Path(path)''',
    '''@contextmanager
def account_lock(book: str, timeout: float = ACCOUNT_LOCK_TIMEOUT_S,
                 path: str | os.PathLike | None = None):
    if path is None:
        # per-venue lock (PAPER.md §12.6): demo and prod are DIFFERENT exchange accounts —
        # serializing them against each other would be pure false contention
        from convexity.paper.venue import state_root
        path = state_root() / "account.lock"
    p = pathlib.Path(path)''',
    "V3b lock resolve")

# ---- V4: manual ledger/sync paths resolve per venue --------------------------------------------
patch("src/convexity/paper/manual.py",
    '''MANUAL_RESERVE_PATH = "data/paper/manual_reserve.json"
SYNC_STATE_PATH = "data/paper/manual_sync.json"''',
    '''MANUAL_RESERVE_PATH = "data/paper/manual_reserve.json"  # demo; prod resolves via venue
SYNC_STATE_PATH = "data/paper/manual_sync.json"


def reserve_path() -> pathlib.Path:
    from convexity.paper.venue import state_root

    return state_root() / "manual_reserve.json"


def sync_state_path() -> pathlib.Path:
    from convexity.paper.venue import state_root

    return state_root() / "manual_sync.json"''',
    "V4a manual path helpers")

patch("src/convexity/paper/manual.py",
    '''def load_reserve(path: str | pathlib.Path = MANUAL_RESERVE_PATH) -> dict[str, float]:
    p = pathlib.Path(path)''',
    '''def load_reserve(path: str | pathlib.Path | None = None) -> dict[str, float]:
    p = pathlib.Path(path) if path is not None else reserve_path()''',
    "V4b load_reserve")

patch("src/convexity/paper/manual.py",
    '''def save_reserve(reserve: dict[str, float],
                 path: str | pathlib.Path = MANUAL_RESERVE_PATH) -> None:
    p = pathlib.Path(path)''',
    '''def save_reserve(reserve: dict[str, float],
                 path: str | pathlib.Path | None = None) -> None:
    p = pathlib.Path(path) if path is not None else reserve_path()''',
    "V4c save_reserve")

patch("src/convexity/paper/manual.py",
    '''def load_sync_state(now_ms: int,
                    path: str | pathlib.Path = SYNC_STATE_PATH) -> dict:''',
    '''def load_sync_state(now_ms: int,
                    path: str | pathlib.Path | None = None) -> dict:''',
    "V4d load_sync sig")

patch("src/convexity/paper/manual.py",
    '''    anchor that keeps every pre-feature order (none of which carries the ``cvx-`` stamp) out of
    attribution forever."""
    p = pathlib.Path(path)''',
    '''    anchor that keeps every pre-feature order (none of which carries the ``cvx-`` stamp) out of
    attribution forever."""
    p = pathlib.Path(path) if path is not None else sync_state_path()''',
    "V4e load_sync resolve")

patch("src/convexity/paper/manual.py",
    '''def _save_sync_state(state: dict, path: str | pathlib.Path = SYNC_STATE_PATH) -> None:
    p = pathlib.Path(path)''',
    '''def _save_sync_state(state: dict, path: str | pathlib.Path | None = None) -> None:
    p = pathlib.Path(path) if path is not None else sync_state_path()''',
    "V4f save_sync resolve")

patch("src/convexity/paper/manual.py",
    '''    ledger_path: str | pathlib.Path = MANUAL_RESERVE_PATH,
    state_path: str | pathlib.Path = SYNC_STATE_PATH,
) -> tuple[list[dict], list[str]]:''',
    '''    ledger_path: str | pathlib.Path | None = None,
    state_path: str | pathlib.Path | None = None,
) -> tuple[list[dict], list[str]]:''',
    "V4g auto_sync sig")

# ---- V5: cli uses the resolved paths ------------------------------------------------------------
CLI = "src/convexity/paper/cli.py"

patch(CLI, '''from convexity.paper.books import (
    BOOKS,
    BookSpec,
    assert_disjoint_domains,
    book_domain,
    book_filters,
)''',
    '''from convexity.paper.books import (
    BOOKS,
    BookSpec,
    assert_disjoint_domains,
    book_domain,
    book_filters,
    book_journal_path,
)''',
    "V5a import")

patch(CLI, '''def _journal(spec: BookSpec) -> Journal:
    return Journal(spec.journal_path)''',
    '''def _journal(spec: BookSpec) -> Journal:
    return Journal(book_journal_path(spec))''',
    "V5b _journal")

patch(CLI, '''def _kill_path(spec: BookSpec) -> pathlib.Path:
    return pathlib.Path(f"data/paper/{spec.name}/KILL")''',
    '''def _kill_path(spec: BookSpec) -> pathlib.Path:
    from convexity.paper.venue import state_root

    return state_root() / spec.name / "KILL"''',
    "V5c kill path")

patch(CLI, '''    others = {b.name: Journal(b.journal_path).last("allocate") for b in other}''',
    '''    others = {b.name: Journal(book_journal_path(b)).last("allocate") for b in other}''',
    "V5d allocate others")

patch(CLI, '''    print(f"[status] book={spec.name} journal={spec.journal_path}")''',
    '''    from convexity.paper.venue import venue
    print(f"[status] book={spec.name} venue={venue()} journal={book_journal_path(spec)}")''',
    "V5e status")

patch(CLI, '''    from convexity.paper.manual import (
        MANUAL_RESERVE_PATH,
        load_reserve,
        load_sync_state,
    )''',
    '''    from convexity.paper.manual import load_reserve, load_sync_state, reserve_path''',
    "V5f cmd_manual imports")

patch(CLI, '''        print(f"[manual] no operator reserves ({MANUAL_RESERVE_PATH}) — auto-adopted from "
              f"non-cvx fills at every sync")''',
    '''        print(f"[manual] no operator reserves ({reserve_path()}) — auto-adopted from "
              f"non-cvx fills at every sync")''',
    "V5g cmd_manual path")

patch(CLI, '''            live_books = [
                b.name for b in BOOKS.values()
                if any(r.get("mode") == "live" for r in _journal(b).read("targets"))]''',
    '''            live_books = [
                b.name for b in BOOKS.values()
                if any(r.get("mode") == "live" for r in _journal(b).read("targets"))]
            # (this venue's journals only — a demo book's live targets never block a prod
            # bootstrap; the state roots are disjoint by construction, PAPER.md §12.6)''',
    "V5h bootstrap comment")

# ---- V6: watchdog + preflight venue-resolved ----------------------------------------------------
patch("scripts/fleet_watchdog.py",
    '''from convexity.paper.books import BOOKS  # noqa: E402''',
    '''from convexity.paper.books import BOOKS, book_journal_path  # noqa: E402''',
    "V6a watchdog import")

patch("scripts/fleet_watchdog.py",
    '''        j = Journal(spec.journal_path)''',
    '''        j = Journal(book_journal_path(spec))''',
    "V6b watchdog path")

patch("scripts/prod_preflight.py",
    '''    pp = pathlib.Path("data/paper")
    rec("PASS" if pp.is_dir() and not pp.is_symlink() else "FAIL", "paper-state",
        "own real dir (journals isolated from demo)" if pp.is_dir() and not pp.is_symlink()
        else "data/paper must be a REAL dir, not shared with demo")''',
    '''    from convexity.paper.venue import state_root
    pp = state_root()
    pp.mkdir(parents=True, exist_ok=True)
    ok_state = pp.is_dir() and not pp.is_symlink() and pp.name != "paper"
    rec("PASS" if ok_state else "FAIL", "paper-state",
        f"venue state root {pp} (prod journals isolated from demo's data/paper)" if ok_state
        else f"state root resolves to {pp} — prod must namespace under data/prod "
             f"(is CONVEXITY_VENUE=prod set?)")''',
    "V6c preflight state root")

patch("scripts/prod_preflight.py",
    '''    from convexity.paper.books import BOOKS
    from convexity.paper.journal import Journal
    allocs = {}
    for name, spec in BOOKS.items():
        a = Journal(spec.journal_path).last("allocate")
        allocs[name] = float(a["amount"]) if a else None''',
    '''    from convexity.paper.books import BOOKS, book_journal_path
    from convexity.paper.journal import Journal
    allocs = {}
    for name, spec in BOOKS.items():
        a = Journal(book_journal_path(spec)).last("allocate")
        allocs[name] = float(a["amount"]) if a else None''',
    "V6d preflight allocs")

patch("scripts/prod_preflight.py",
    '''    live_targets = any(
        any(r.get("mode") == "live" for r in Journal(s.journal_path).read("targets"))
        for s in BOOKS.values())''',
    '''    live_targets = any(
        any(r.get("mode") == "live" for r in Journal(book_journal_path(s)).read("targets"))
        for s in BOOKS.values())''',
    "V6e preflight targets")

# ---- V7: backup covers both venues --------------------------------------------------------------
patch("scripts/journal_backup.sh",
    '''  data/paper/journal.jsonl \\
  data/paper/commodity/journal.jsonl \\
  data/paper/equity/journal.jsonl \\
  data/paper/manual_reserve.json \\
  data/paper/manual_sync.json \\''',
    '''  data/paper/journal.jsonl \\
  data/paper/commodity/journal.jsonl \\
  data/paper/equity/journal.jsonl \\
  data/paper/manual_reserve.json \\
  data/paper/manual_sync.json \\
  data/prod/crypto/journal.jsonl \\
  data/prod/commodity/journal.jsonl \\
  data/prod/equity/journal.jsonl \\
  data/prod/manual_reserve.json \\
  data/prod/manual_sync.json \\''',
    "V7 backup both venues")

# ---- V8: prod units (committed under deploy/prod/) ----------------------------------------------
SHIFT = {"00:10:00": "00:15:00", "00:25:00": "00:30:00", "00:40:00": "00:45:00",
         "00:45:00": "00:50:00", "01:05:00": "01:10:00", "01:10:00": "01:15:00",
         "02:20:00": "02:25:00", "02:50:00": "02:55:00", "03:20:00": "03:25:00",
         "12:00:00": "12:10:00"}
UNITS = ("paper-rebalance", "paper-finalize", "paper-recon", "commodity-rebalance",
         "commodity-finalize", "commodity-recon", "equity-rebalance", "equity-finalize",
         "equity-recon", "watchdog")
outdir = repo / "deploy/prod"
outdir.mkdir(parents=True, exist_ok=True)
for u in UNITS:
    for ext in ("service", "timer"):
        src = repo / "deploy" / f"convexity-{u}.{ext}"
        if not src.exists():
            continue
        lines = []
        for ln in src.read_text().splitlines():
            if ln.startswith("Description="):
                ln = ln + " (PROD)"
            elif ln.startswith("ExecStart="):
                # venue pinned ON the command line: EnvironmentFile= overrides Environment=,
                # so this is the ONLY placement .env can never override (PAPER.md §12.6)
                ln = ln.replace("ExecStart=", "ExecStart=/usr/bin/env CONVEXITY_VENUE=prod ", 1)
            elif ln.startswith("OnCalendar="):
                for old_t, new_t in SHIFT.items():
                    if old_t in ln:
                        ln = ln.replace(old_t, new_t)
                        break
            lines.append(ln)
        (outdir / f"convexity-prod-{u}.{ext}").write_text("\n".join(lines) + "\n")
        applied.append(f"unit:convexity-prod-{u}.{ext}")

# ---- V9: tests -----------------------------------------------------------------------------------
tp = repo / "tests/paper/test_fleet_infra.py"
t = tp.read_text()
assert "test_venue_state_namespacing" not in t
t = t.rstrip() + '''


def test_venue_state_namespacing(monkeypatch):
    """PAPER.md §12.6: ONE checkout, both venues — demo state paths byte-identical to the
    legacy layout; prod isolates every stateful path under data/prod."""
    from convexity.paper import venue as V
    from convexity.paper.books import BOOKS, book_journal_path
    from convexity.paper.manual import reserve_path, sync_state_path

    monkeypatch.delenv("CONVEXITY_VENUE", raising=False)
    assert str(V.state_root()) == "data/paper"
    assert book_journal_path(BOOKS["crypto"]) == "data/paper/journal.jsonl"  # legacy preserved
    assert book_journal_path(BOOKS["equity"]) == "data/paper/equity/journal.jsonl"
    assert str(reserve_path()) == "data/paper/manual_reserve.json"

    monkeypatch.setenv("CONVEXITY_VENUE", "prod")
    assert str(V.state_root()) == "data/prod"
    assert book_journal_path(BOOKS["crypto"]) == "data/prod/crypto/journal.jsonl"
    assert book_journal_path(BOOKS["commodity"]) == "data/prod/commodity/journal.jsonl"
    assert book_journal_path(BOOKS["equity"]) == "data/prod/equity/journal.jsonl"
    assert str(reserve_path()) == "data/prod/manual_reserve.json"
    assert str(sync_state_path()) == "data/prod/manual_sync.json"


def test_venue_lock_isolation(tmp_path, monkeypatch):
    """Demo and prod are different exchange accounts: their locks must not contend."""
    import os

    from convexity.paper.lock import account_lock

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONVEXITY_VENUE", raising=False)
    with account_lock("demo-test", timeout=1.0):
        monkeypatch.setenv("CONVEXITY_VENUE", "prod")
        with account_lock("prod-test", timeout=1.0):  # would deadlock if shared
            assert os.path.exists("data/paper/account.lock")
            assert os.path.exists("data/prod/account.lock")
''' + "\n"
tp.write_text(t)
applied.append("tests")

# ---- V10: docs -----------------------------------------------------------------------------------
dp = repo / "docs/specs/PAPER.md"
t = dp.read_text()
old = "**Prod cutover runbook** (separate-checkout deployment — `scripts/setup_prod_checkout.sh`" \
      " creates `/home/speculation/convexity-prod` with own journals/lock/.env/venv, shared " \
      "read-only lake, prod-named units installed DISABLED at +5 min stagger; " \
      "`scripts/prod_preflight.py` is the read-only gate): "
new = "**Prod cutover runbook** (SINGLE-CHECKOUT dual-venue deployment, owner decision " \
      "2026-07-10 — supersedes the separate-checkout design): both key sets live in the ONE " \
      ".env; `CONVEXITY_VENUE` must NOT be set there (EnvironmentFile= overrides Environment=)" \
      " — code defaults demo, prod units pin `/usr/bin/env CONVEXITY_VENUE=prod` on ExecStart. " \
      "Paper state is venue-namespaced: demo keeps `data/paper/**` untouched; prod gets " \
      "`data/prod/<book>/journal.jsonl` + own account.lock/manual ledgers (fresh accounting " \
      "chain, no demo-history bleed). Prod units live in `deploy/prod/` (+5 min stagger) so " \
      "BOTH fleets run concurrently — demo stays the shadow book. " \
      "`scripts/prod_preflight.py` (run with CONVEXITY_VENUE=prod) is the gate: "
assert old in t, "runbook anchor drifted"
dp.write_text(t.replace(old, new))
applied.append("PAPER.md")

rp = repo / "deploy/README.md"
t = rp.read_text()
old = "The prod fleet runs from `/home/speculation/convexity-prod`"
assert old in t
t = t.replace("""## Prod deployment (separate checkout, PAPER.md §12.6)

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
`/home/speculation/backups/<checkout>` and per-checkout `journal-backups-<checkout>` branches.""",
"""## Prod deployment (single checkout, dual venue — PAPER.md §12.6)

Both fleets run from THIS checkout. The one `.env` holds BOTH key sets
(`BINANCE_FUTURES_DEMO_API_KEY/_SECRET` and `BINANCE_FUTURES_API_KEY/_SECRET`) and must NOT set
`CONVEXITY_VENUE` (EnvironmentFile= overrides Environment=; code defaults to demo). Prod units
(`deploy/prod/`, +5 min stagger) pin the venue on ExecStart: `/usr/bin/env CONVEXITY_VENUE=prod`.
State is venue-namespaced: demo at `data/paper/**` (unchanged), prod at
`data/prod/<book>/journal.jsonl` + own account.lock/manual ledgers. One backup unit tars both.

```bash
CONVEXITY_VENUE=prod .venv/bin/python scripts/prod_preflight.py   # gate; exit 0 = ready
sudo cp deploy/prod/convexity-prod-*.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload
```

Cutover (first cycle SUPERVISED, §12.6): preflight green -> `CONVEXITY_VENUE=prod ... manual
--bootstrap` (if you hold positions) -> `CONVEXITY_VENUE=prod ... allocate-split --total <prod
USDT>` -> run one prod rebalance/finalize by hand watching the `venue` record and `finalized
mode=ids` -> `sudo systemctl enable --now convexity-prod-{paper,commodity,equity}-{rebalance,finalize,recon}.timer convexity-prod-watchdog.timer`.
Manual prod CLI runs always prefix `CONVEXITY_VENUE=prod`; bare commands are demo by default.""")
rp.write_text(t)
applied.append("README")
print("ALL APPLIED:", len(applied), "->", applied[:8], "...")
