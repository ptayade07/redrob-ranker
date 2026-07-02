"""Stage 5 sanity check -- run honeypot detection against
sample_candidates.json (per the original plan) and, if available, the full
dataset (to confirm the real hit rate/coverage against the spec's ~80).

Usage:
    python scripts/check_honeypots.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.honeypots import check_honeypot
from redrob_ranker.data_loader import iter_candidates

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "data" / "reference" / "sample_candidates.json"
FULL_PATH = REPO_ROOT / "data" / "raw" / "candidates.jsonl"


def main():
    candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print("=" * 80)
    print(f"SAMPLE ({len(candidates)} candidates from data/reference/sample_candidates.json)")
    print("=" * 80)
    flagged = 0
    for c in candidates:
        result = check_honeypot(c)
        if result.is_honeypot:
            flagged += 1
            print(f"{result.candidate_id}: FLAGGED")
            for r in result.reasons:
                print(f"   - {r}")
    if flagged == 0:
        print("No honeypots flagged in the 50-candidate sample.")
        print("Expected: spec says ~80/100,000 honeypots (~0.08%), so the expected")
        print("count in a 50-candidate sample is ~0.04 -- zero is the normal outcome,")
        print("not a sign the detector is broken (Stage 2 confirmed this same thing).")

    if FULL_PATH.exists():
        print()
        print("=" * 80)
        print("FULL DATASET")
        print("=" * 80)
        t0 = time.time()
        n = 0
        n_flagged = 0
        examples = []
        for c in iter_candidates(FULL_PATH):
            n += 1
            result = check_honeypot(c)
            if result.is_honeypot:
                n_flagged += 1
                if len(examples) < 5:
                    examples.append(result)
        elapsed = time.time() - t0
        print(f"scanned {n} candidates in {elapsed:.1f}s")
        print(f"honeypots flagged: {n_flagged} ({n_flagged / n * 100:.2f}% of the pool)")
        print(f"spec's stated honeypot count: ~80 ({80 / n * 100:.2f}% of the pool)")
        print()
        print("first 5 flagged examples:")
        for r in examples:
            print(f"  {r.candidate_id}: {r.reasons}")
    else:
        print()
        print(f"Full dataset not found at {FULL_PATH.relative_to(REPO_ROOT)} -- skipping full-scan stats.")


if __name__ == "__main__":
    main()
