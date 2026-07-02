"""
Streaming reader for candidates.jsonl.

Reads one JSON object per line rather than json.load()-ing the whole
~490MB / 100,000-row file at once. At 100K candidates this keeps peak
memory well under the 16GB budget (submission_spec.md section 3) and lets
the pipeline process candidates as a stream instead of holding two full
copies (raw JSON + parsed) in memory simultaneously.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_candidates(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_candidates(path: str | Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n
