"""Load the trading universe from candidates.md (the machine-readable ```json block)."""

from __future__ import annotations

import json
import pathlib
import re

_UNIVERSE_MD = pathlib.Path(__file__).with_name("candidates.md")


def load_universe(path: str | pathlib.Path | None = None) -> dict:
    """Return the parsed universe dict: keys `meta`, `crypto`, `equities`, `commodities`."""
    text = pathlib.Path(path or _UNIVERSE_MD).read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    if not m:
        raise ValueError("no ```json block found in candidates.md")
    return json.loads(m.group(1))


def all_symbols(sleeves: tuple[str, ...] = ("crypto", "equities", "commodities")) -> list[str]:
    """Flat list of Binance perp symbols across the requested sleeves."""
    u = load_universe()
    return [d["symbol"] for s in sleeves for d in u[s]]
