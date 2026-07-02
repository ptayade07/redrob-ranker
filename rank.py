#!/usr/bin/env python
"""
Single entry point: candidates.jsonl -> top-100 submission CSV.

    python rank.py --candidates ./data/raw/candidates.jsonl --out ./output/submission.csv

No network calls, no GPU -- see src/redrob_ranker/scoring.py and
narrative.py for why (submission_spec.md section 3 compute constraints).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.data_loader import iter_candidates  # noqa: E402
from redrob_ranker.scoring import score_candidates  # noqa: E402

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
TOP_N = 100

# Runtime budget is 5 minutes (submission_spec.md section 3). Warn loudly
# if we're within a minute of it, rather than only reporting pass/fail --
# a run that "passes" at 280s has no safety margin for a slower grading
# machine or a slightly larger future dataset.
RUNTIME_WARNING_THRESHOLD_SECONDS = 240


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates against the Redrob JD and write the top-100 submission CSV."
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV")
    args = parser.parse_args()

    t_start = time.time()

    print(f"Loading candidates from {args.candidates} ...")
    t0 = time.time()
    candidates = list(iter_candidates(args.candidates))
    print(f"  loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    print("Scoring (honeypot gate, feature extraction, narrative similarity, "
          "composite scoring, reasoning generation) ...")
    t0 = time.time()
    scored, honeypots_excluded = score_candidates(candidates)
    print(f"  scored {len(scored)} candidates ({len(honeypots_excluded)} honeypots "
          f"excluded) in {time.time() - t0:.1f}s")

    top = scored[:TOP_N]
    if len(top) < TOP_N:
        raise RuntimeError(f"Only {len(top)} candidates available after honeypot "
                            f"exclusion; need {TOP_N}.")

    # Round for display, then re-sort by (rounded_score desc, candidate_id
    # asc) -- guarantees validate_submission.py's tie-break rule holds even
    # where rounding creates ties that didn't exist at full float precision
    # (score_candidates() only guarantees the tie-break on exact scores).
    rows = [(s.candidate_id, round(s.final_score, 6), s.reasoning) for s in top]
    rows.sort(key=lambda r: (-r[1], r[0]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_HEADER)
        for rank, (cid, score, reasoning) in enumerate(rows, start=1):
            writer.writerow([cid, rank, score, reasoning])

    total_elapsed = time.time() - t_start
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Total wall-clock time: {total_elapsed:.1f}s (budget: 300s)")
    if total_elapsed > RUNTIME_WARNING_THRESHOLD_SECONDS:
        print(f"WARNING: runtime {total_elapsed:.1f}s is within "
              f"{300 - RUNTIME_WARNING_THRESHOLD_SECONDS}s of the 5-minute budget. "
              "This needs a closer look before submission, not just a pass/fail check.")


if __name__ == "__main__":
    main()
