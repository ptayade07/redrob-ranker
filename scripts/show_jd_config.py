"""Stage 3 sanity check -- print the structured JD config for review.

Usage:
    python scripts/show_jd_config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.config import JD


def main():
    print("=" * 80)
    print("MUST-HAVE SKILL BUCKETS")
    print("=" * 80)
    print("retrieval_ranking_skills:", JD.retrieval_ranking_skills)
    print("vector_db_skills:        ", JD.vector_db_skills)
    print("core_python_ml_skills:   ", JD.core_python_ml_skills)
    print("eval_methodology_terms:  ", JD.eval_methodology_terms)

    print()
    print("=" * 80)
    print("NICE-TO-HAVE SKILL BUCKETS")
    print("=" * 80)
    print("fine_tuning_skills:  ", JD.fine_tuning_skills)
    print("infra_mlops_skills:  ", JD.infra_mlops_skills)
    print("general_data_skills: ", JD.general_data_skills)

    print()
    print("=" * 80)
    print("NEGATIVE SIGNAL BUCKETS")
    print("=" * 80)
    print("cv_speech_robotics_skills:", JD.cv_speech_robotics_skills)
    print("framework_only_markers:   ", JD.framework_only_markers)
    print("consulting_firms:         ", JD.consulting_firms)

    print()
    print("=" * 80)
    print("LOCATION TIERS")
    print("=" * 80)
    for t in JD.location_tiers:
        print(f"  tier {t.tier} ({t.label}), weight={t.weight}: {t.cities}")
    print(f"  outside India: base_weight={JD.non_india_base_weight}, "
          f"relocate_weight={JD.non_india_relocate_weight}")

    print()
    print("=" * 80)
    print("NOTICE PERIOD / EXPERIENCE BAND")
    print("=" * 80)
    print("ideal_notice_period_days:      ", JD.ideal_notice_period_days)
    print("steep_penalty_notice_period_days:", JD.steep_penalty_notice_period_days)
    print("experience_band_years:         ", JD.experience_band_years)
    print("experience_sweet_spot_years:   ", JD.experience_sweet_spot_years)

    print()
    print("=" * 80)
    print("DISQUALIFIERS")
    print("=" * 80)
    for d in JD.disqualifiers:
        print(f"[{d.key}]")
        print(f"  JD quote: {d.jd_quote}")
        print(f"  implemented in: {d.implemented_in}")
        print()

    print("=" * 80)
    print("IDEAL CANDIDATE NARRATIVE (Stage 6 text-similarity anchor)")
    print("=" * 80)
    print(JD.ideal_candidate_narrative)


if __name__ == "__main__":
    main()
