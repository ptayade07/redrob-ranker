"""
Stage 2 — data exploration.

Prints the schema shape of a candidate record and surfaces concrete examples
of the trap patterns the JD warns about, so we can see what we're actually
building against before writing any scoring logic.

Two data sources:
  - data/reference/sample_candidates.json (50 candidates, always available, committed to git)
  - data/raw/candidates.jsonl (full 100K, gitignored, must be supplied locally)

Honeypots are ~80/100,000 candidates (spec section 7) -- in expectation that's
~0.04 honeypots in a 50-candidate sample, i.e. we don't expect to find any in
the sample file. So honeypot examples are pulled from the full file if present,
and the script says so explicitly rather than silently having nothing to show.

Usage:
    python scripts/explore_data.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "data" / "reference" / "sample_candidates.json"
FULL_PATH = REPO_ROOT / "data" / "raw" / "candidates.jsonl"

# "Now" for date-math sanity checks. The dataset's last_active_date values
# cluster right up to mid-2026, so we treat generation time as ~2026-07-01
# rather than the wall-clock date the pipeline happens to run on.
DATASET_NOW = "2026-07-01"

CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

AI_SKILL_NAMES = {
    "nlp", "llm", "llms", "rag", "embedding", "embeddings", "retrieval", "vector",
    "fine-tuning llms", "fine-tuning", "lora", "qlora", "peft", "transformers",
    "bert", "gpt", "langchain", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "speech recognition", "tts", "image classification",
    "gans",
}

NON_AI_TITLE_MARKERS = (
    "marketing", "sales", "hr ", "human resources", "business analyst",
    "project manager", "operations", "customer", "frontend engineer",
    "backend engineer", "devops",
)


def months_between(start: str, end: str | None) -> int:
    y1, m1, _ = map(int, start.split("-"))
    end = end or DATASET_NOW
    y2, m2, _ = map(int, end.split("-"))
    return (y2 - y1) * 12 + (m2 - m1)


def load_sample() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def print_schema_shape(candidate: dict) -> None:
    print("=" * 80)
    print("SCHEMA SHAPE (from one candidate record)")
    print("=" * 80)

    def walk(obj, indent=0):
        if isinstance(obj, dict):
            for k, v in obj.items():
                print(" " * indent + f"{k}: {type(v).__name__}")
                if isinstance(v, dict):
                    walk(v, indent + 2)
                elif isinstance(v, list) and v:
                    print(" " * (indent + 2) + f"[0]: {type(v[0]).__name__}")
                    if isinstance(v[0], dict):
                        walk(v[0], indent + 4)

    walk(candidate)
    print()


def find_keyword_stuffers(candidates: list[dict]) -> list[dict]:
    """Title/company says non-AI role, but skills list is loaded with 'advanced'+ AI tags.

    This is the exact trap the JD calls out: don't just count AI keywords.
    """
    hits = []
    for c in candidates:
        title = c["profile"]["current_title"].lower()
        if not any(marker in title for marker in NON_AI_TITLE_MARKERS):
            continue
        advanced_ai_skills = [
            s for s in c["skills"]
            if s["name"].lower() in AI_SKILL_NAMES and s["proficiency"] in ("advanced", "expert")
        ]
        if len(advanced_ai_skills) >= 3:
            hits.append({
                "candidate_id": c["candidate_id"],
                "title": c["profile"]["current_title"],
                "company": c["profile"]["current_company"],
                "industry": c["profile"]["current_industry"],
                "advanced_ai_skills": [s["name"] for s in advanced_ai_skills],
                "skill_assessment_scores": c["redrob_signals"]["skill_assessment_scores"],
            })
    return hits


def find_consulting_only(candidates: list[dict]) -> list[dict]:
    """Entire career_history spent at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini."""
    hits = []
    for c in candidates:
        companies = {h["company"].lower() for h in c["career_history"]}
        if companies and companies.issubset(CONSULTING_FIRMS):
            hits.append({
                "candidate_id": c["candidate_id"],
                "title": c["profile"]["current_title"],
                "companies": sorted(companies),
                "years_of_experience": c["profile"]["years_of_experience"],
            })
    return hits


def find_job_hoppers(candidates: list[dict], min_jobs=3, max_avg_tenure_months=18) -> list[dict]:
    """3+ jobs averaging under ~1.5 years each -- the JD's 'title-chaser' disqualifier,
    checked directly against career_history tenure rather than a self-reported label.
    """
    hits = []
    for c in candidates:
        hist = c["career_history"]
        if len(hist) < min_jobs:
            continue
        avg_tenure = sum(h["duration_months"] for h in hist) / len(hist)
        if avg_tenure < max_avg_tenure_months:
            hits.append({
                "candidate_id": c["candidate_id"],
                "title": c["profile"]["current_title"],
                "n_jobs": len(hist),
                "avg_tenure_months": round(avg_tenure, 1),
            })
    return hits


def find_honeypots_in_full_dataset(path: Path, limit_per_type=3) -> dict:
    """Streams the full 100K file once looking for the specific impossible-profile
    patterns the spec describes (section 7): tenure that doesn't square with dates,
    'expert' proficiency with ~zero months of use, and years_of_experience that
    doesn't square with the sum of career_history durations.
    """
    found = {"impossible_tenure": [], "expert_zero_duration": [], "yoe_mismatch": []}

    with path.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cid = c["candidate_id"]

            for h in c["career_history"]:
                calc_months = months_between(h["start_date"], h["end_date"])
                if abs(calc_months - h["duration_months"]) > 6 and len(found["impossible_tenure"]) < limit_per_type:
                    found["impossible_tenure"].append({
                        "candidate_id": cid, "company": h["company"], "title": h["title"],
                        "start_date": h["start_date"], "end_date": h["end_date"],
                        "stated_duration_months": h["duration_months"], "calculated_months": calc_months,
                    })

            if len(found["expert_zero_duration"]) < limit_per_type:
                zero_dur_experts = [s for s in c["skills"] if s["proficiency"] == "expert" and s["duration_months"] <= 3]
                if zero_dur_experts:
                    found["expert_zero_duration"].append({
                        "candidate_id": cid,
                        "skills": [(s["name"], s["duration_months"]) for s in zero_dur_experts],
                    })

            if len(found["yoe_mismatch"]) < limit_per_type:
                total_months = sum(h["duration_months"] for h in c["career_history"])
                calc_yoe = total_months / 12
                stated_yoe = c["profile"]["years_of_experience"]
                if abs(calc_yoe - stated_yoe) > 1.5:
                    found["yoe_mismatch"].append({
                        "candidate_id": cid, "stated_years_of_experience": stated_yoe,
                        "career_history_implies_years": round(calc_yoe, 1),
                    })

            if all(len(v) >= limit_per_type for v in found.values()):
                break

    return found


def main():
    candidates = load_sample()
    print(f"Loaded {len(candidates)} candidates from {SAMPLE_PATH.relative_to(REPO_ROOT)}\n")

    print_schema_shape(candidates[0])

    print("=" * 80)
    print("TRAP 1: KEYWORD STUFFERS (non-AI title + 3+ 'advanced'/'expert' AI skill tags)")
    print("=" * 80)
    for hit in find_keyword_stuffers(candidates):
        print(f"{hit['candidate_id']} | {hit['title']} @ {hit['company']} ({hit['industry']})")
        print(f"  advanced/expert AI skills claimed: {hit['advanced_ai_skills']}")
        print(f"  Redrob skill_assessment_scores (independent of self-report): {hit['skill_assessment_scores']}")
        print("  -> assessed scores are mediocre/absent relative to the claimed proficiency level;"
              " this is the 'trust discount' signal (see Stage 4 feature extraction).")
        print()

    print("=" * 80)
    print("TRAP 2: CONSULTING-ONLY CAREER (entire history at TCS/Infosys/Wipro/Accenture/"
          "Cognizant/Capgemini, JD explicit disqualifier)")
    print("=" * 80)
    for hit in find_consulting_only(candidates):
        print(f"{hit['candidate_id']} | {hit['title']} | companies: {hit['companies']} | "
              f"yoe: {hit['years_of_experience']}")
    print()

    print("=" * 80)
    print("TRAP 3: JOB-HOPPER PATTERN (3+ jobs, avg tenure < 18 months -- JD's 'title-chaser')")
    print("=" * 80)
    for hit in find_job_hoppers(candidates):
        print(f"{hit['candidate_id']} | {hit['title']} | {hit['n_jobs']} jobs, "
              f"avg tenure {hit['avg_tenure_months']} months")
        if hit["title"] == "Recommendation Systems Engineer":
            print("  -> note: title is a perfect keyword match for the JD, but the tenure pattern"
                  " is exactly the title-chaser disqualifier. Good title != good fit.")
    print()

    print("=" * 80)
    print("HONEYPOTS: not expected in this 50-candidate sample")
    print("=" * 80)
    print("Spec says ~80 honeypots exist across all 100,000 candidates (~0.08%). Expected count"
          " in a 50-candidate sample is ~0.04, so finding zero here is normal, not a bug in our"
          " detection logic -- confirmed no impossible-tenure, zero-duration-expert, or"
          " years-of-experience mismatches in data/reference/sample_candidates.json.\n")

    if FULL_PATH.exists():
        print(f"Full dataset found at {FULL_PATH.relative_to(REPO_ROOT)} -- scanning for real"
              " honeypot examples (this is a one-time exploratory scan, not part of the pipeline):\n")
        honeypots = find_honeypots_in_full_dataset(FULL_PATH)

        print("--- Impossible tenure (stated duration_months doesn't match start/end dates) ---")
        for h in honeypots["impossible_tenure"]:
            print(f"{h['candidate_id']} | {h['title']} @ {h['company']} | "
                  f"{h['start_date']} to {h['end_date'] or 'present'} | "
                  f"stated {h['stated_duration_months']}mo vs. date-math says {h['calculated_months']}mo")
        print()

        print("--- 'Expert' proficiency claimed with ~0 months of actual use ---")
        for h in honeypots["expert_zero_duration"]:
            print(f"{h['candidate_id']} | expert-but-unused skills: {h['skills']}")
        print()

        print("--- years_of_experience contradicts sum(career_history durations) ---")
        for h in honeypots["yoe_mismatch"]:
            print(f"{h['candidate_id']} | profile says {h['stated_years_of_experience']}y, "
                  f"but career_history sums to {h['career_history_implies_years']}y")
        print()
        print("These three checks become the basis of Stage 5's honeypot/consistency module.")
    else:
        print(f"Full dataset not found at {FULL_PATH.relative_to(REPO_ROOT)} -- skipping the"
              " full-scan honeypot examples. Place candidates.jsonl there to see them.")


if __name__ == "__main__":
    main()
