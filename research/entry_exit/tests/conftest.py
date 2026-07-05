"""Make the entry_exit package importable when running these tests directly:
    .venv/bin/python -m pytest research/entry_exit/tests -q
(research/ is not on convexity's pythonpath; testpaths=["tests"] never collects this).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # research/
