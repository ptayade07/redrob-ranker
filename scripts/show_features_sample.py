"""Stage 4 sanity check -- extract features for the 50 sample candidates
and print them as a table, plus a close-up on the trap candidates Stage 2
identified, so we can see the features actually separate traps from fits
before Stage 5/6 build on top of them.

Usage:
    python scripts/show_features_sample.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.features import extract_features
from redrob_ranker.narrative import NarrativeSimilarityScorer, candidate_narrative_text
from redrob_ranker.config import JD

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "data" / "reference" / "sample_candidates.json"

# Trap candidates identified in Stage 2 (scripts/explore_data.py), for a
# targeted before/after look at how the features score them.
TRAP_CANDIDATE_IDS = {
    "CAND_0000001": "keyword stuffer (Backend Engineer, advanced AI skills, low assessments)",
    "CAND_0000014": "keyword stuffer (Frontend Engineer, advanced FAISS/OpenSearch/GANs)",
    "CAND_0000003": "consulting-only (TCS, Customer Support)",
    "CAND_0000024": "consulting-only (Infosys+TCS, HR Manager)",
    "CAND_0000031": "job hopper w/ perfect title (Recommendation Systems Engineer, 4 jobs)",
}


def main():
    candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    features = [extract_features(c) for c in candidates]

    # Narrative similarity needs a corpus fit. Stage 6/7 will fit this on
    # the full 100K candidates; here we fit on the 50-sample just to sanity
    # check the mechanism works and to see relative ordering.
    texts = [f.narrative_text for f in features]
    scorer = NarrativeSimilarityScorer(JD.ideal_candidate_narrative).fit(texts)
    sims = scorer.score_many(texts)
    for f, sim in zip(features, sims):
        f.narrative_similarity = round(float(sim), 4)

    rows = []
    for f in features:
        rows.append({
            "candidate_id": f.candidate_id,
            "title": f.current_title,
            "company": f.current_company,
            "yoe": f.years_of_experience,
            "retrieval_cov": round(f.retrieval_ranking_coverage, 2),
            "vecdb_cov": round(f.vector_db_coverage, 2),
            "python_cov": round(f.core_python_ml_coverage, 2),
            "cv_speech_cov": round(f.cv_speech_coverage, 2),
            "title_rel": round(f.title_relevance, 2),
            "exp_fit": round(f.experience_fit, 2),
            "loc_fit": round(f.location_fit, 2),
            "notice_fit": round(f.notice_period_fit, 2),
            "narrative_sim": f.narrative_similarity,
            "job_hopper": f.is_job_hopper,
            "consulting_only": f.is_consulting_only,
            "cv_speech_only": f.is_cv_speech_only,
            "recent_llm_only": f.is_recent_llm_only,
            "closed_src_flag": f.is_closed_source_no_validation,
            "prod_evidence": round(f.production_evidence_score, 2),
        })
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("=" * 100)
    print(f"FULL FEATURE TABLE ({len(df)} sample candidates)")
    print("=" * 100)
    print(df.to_string(index=False))

    print()
    print("=" * 100)
    print("SUMMARY STATS")
    print("=" * 100)
    numeric_cols = [
        "retrieval_cov", "vecdb_cov", "python_cov", "cv_speech_cov", "title_rel",
        "exp_fit", "loc_fit", "notice_fit", "narrative_sim", "prod_evidence",
    ]
    print(df[numeric_cols].describe().round(3))

    print()
    print("=" * 100)
    print("TRAP CANDIDATE CLOSE-UP (from Stage 2)")
    print("=" * 100)
    trap_df = df[df["candidate_id"].isin(TRAP_CANDIDATE_IDS)]
    for _, row in trap_df.iterrows():
        print(f"\n{row['candidate_id']} -- {TRAP_CANDIDATE_IDS[row['candidate_id']]}")
        print(f"  title={row['title']!r} @ {row['company']!r}")
        print(f"  retrieval_cov={row['retrieval_cov']} vecdb_cov={row['vecdb_cov']} "
              f"title_rel={row['title_rel']} narrative_sim={row['narrative_sim']}")
        print(f"  flags: job_hopper={row['job_hopper']} consulting_only={row['consulting_only']} "
              f"cv_speech_only={row['cv_speech_only']} recent_llm_only={row['recent_llm_only']}")

    print()
    print("=" * 100)
    print("TOP 5 BY (retrieval_cov + vecdb_cov + narrative_sim), for a directional sanity check")
    print("=" * 100)
    df["_rough_signal"] = df["retrieval_cov"] + df["vecdb_cov"] + df["narrative_sim"]
    top5 = df.sort_values("_rough_signal", ascending=False).head(5)
    print(top5[["candidate_id", "title", "company", "retrieval_cov", "vecdb_cov",
                "narrative_sim", "job_hopper", "consulting_only"]].to_string(index=False))


if __name__ == "__main__":
    main()
