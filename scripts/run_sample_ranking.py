"""Stage 6 sanity check -- score and rank the 50 sample candidates.

IMPORTANT: this is a QUALITATIVE check only ("does the ordering look
sane"), not a validation of final scores. narrative_similarity is
converted to a percentile rank *within the pool being scored* (see
scoring.py's module docstring) -- against 50 candidates that number means
something different than it will against the real 100,000-candidate pool.
score_candidates() emits a RuntimeWarning for exactly this reason; we let
it print rather than silencing it, as a visible reminder every time this
script runs.

Usage:
    python scripts/run_sample_ranking.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.scoring import score_candidates

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "data" / "reference" / "sample_candidates.json"


def main():
    candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    scored, honeypots = score_candidates(candidates)

    print("=" * 100)
    print(f"RANKED SAMPLE ({len(scored)} scored, {len(honeypots)} honeypots excluded)")
    print("=" * 100)
    if honeypots:
        for cid, reasons in honeypots:
            print(f"EXCLUDED (honeypot): {cid} -- {reasons}")
        print()

    for rank, sc in enumerate(scored, start=1):
        f = sc.features
        print(f"#{rank:>2}  score={sc.final_score:.4f}  {sc.candidate_id}  "
              f"{f.current_title} @ {f.current_company} ({f.years_of_experience:g}y, {f.location})")
        print(f"      base_fit={sc.base_fit_score:.3f} logistics={sc.logistics_multiplier:.3f} "
              f"engagement={sc.engagement_multiplier:.3f} risk={sc.risk_multiplier:.3f} "
              f"corroborated={sc.narrative_corroborated}")
        print(f"      reasoning: {sc.reasoning}")
        print()

    print("=" * 100)
    print("SPOT CHECK -- Stage 2/4 trap candidates and where they landed")
    print("=" * 100)
    rank_by_id = {sc.candidate_id: (r, sc) for r, sc in enumerate(scored, start=1)}
    trap_ids = ["CAND_0000001", "CAND_0000014", "CAND_0000003", "CAND_0000024", "CAND_0000031"]
    for cid in trap_ids:
        if cid in rank_by_id:
            r, sc = rank_by_id[cid]
            print(f"{cid}: rank {r}/{len(scored)}, score={sc.final_score:.4f}")
            print(f"   reasoning: {sc.reasoning}")
        else:
            print(f"{cid}: excluded as honeypot")
        print()


if __name__ == "__main__":
    main()
