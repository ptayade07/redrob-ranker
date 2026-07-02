"""
Reasoning-string generation.

submission_spec.md section 3 requires 1-2 sentences grounded in the
candidate's actual profile, connected to specific JD requirements, honest
about gaps, and NOT templated/identical across rows. To satisfy that
without an LLM call (forbidden by the compute budget anyway -- section 3),
this builds the string directly from the same CandidateFeatures values
that drove the score: it can't hallucinate a skill the candidate doesn't
have, because every fact it can cite comes from a field already extracted
from that candidate's own record. Variation across candidates comes for
free from this being fact-driven -- different candidates have different
skills, companies, titles, and numbers, so the sentences differ without
needing an artificial randomizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scoring import ScoredCandidate


def _format_skill_list(names: list[str], limit: int = 3) -> str:
    picks = names[:limit]
    if len(picks) == 1:
        return picks[0]
    return ", ".join(picks[:-1]) + f" and {picks[-1]}"


def _positive_facts(sc: "ScoredCandidate") -> list[str]:
    f = sc.features
    facts = []

    if f.matched_must_have_skills:
        facts.append(f"hands-on experience with {_format_skill_list(f.matched_must_have_skills)}")

    if f.narrative_relevance >= 0.7:
        facts.append("a career narrative that closely matches production ranking/retrieval/recommendation work")
    elif f.narrative_relevance >= 0.4:
        facts.append("career history describing relevant search/ranking/retrieval-flavored work")

    if f.eval_methodology_evidence:
        facts.append("career history referencing evaluation methodology (NDCG/MRR/MAP/A-B testing)")

    if f.matched_nice_to_have_skills:
        facts.append(f"additional exposure to {_format_skill_list(f.matched_nice_to_have_skills, limit=2)}")

    if f.location_fit >= 0.85:
        facts.append(f"a location ({f.location}) matching the JD's preferred locations")

    if f.notice_period_days <= 30:
        facts.append(f"a {f.notice_period_days}-day notice period within the JD's sub-30-day ask")

    if f.engagement_activity_recency >= 0.9 and f.engagement_response_rate >= 0.6:
        facts.append("recently active on the platform with a solid recruiter response rate")

    return facts


def _concern_facts(sc: "ScoredCandidate") -> list[str]:
    f = sc.features
    concerns = []

    if f.is_job_hopper:
        concerns.append(
            f"tenure pattern ({f.n_jobs} roles averaging ~{f.avg_tenure_months:.0f} months each) "
            "matches the JD's title-chaser concern"
        )

    if f.is_consulting_only:
        firms = ", ".join(f.consulting_companies)
        concerns.append(f"entire career has been at consulting firms ({firms}) with no product-company experience")

    if f.is_cv_speech_only:
        concerns.append(
            f"skill background ({_format_skill_list(f.matched_cv_speech_skills)}) is CV/speech-focused "
            "without NLP/IR exposure"
        )

    if f.is_recent_llm_only:
        concerns.append("AI-related skills show only recent, short-duration exposure without earlier production ML history")

    if f.retrieval_ranking_coverage < 0.2 and f.vector_db_coverage < 0.2 and not f.matched_must_have_skills:
        concerns.append("limited direct evidence of retrieval/ranking/vector-DB work in skills or career history")

    if f.location_fit < 0.5:
        concerns.append(f"based in {f.location}, outside the JD's preferred locations")

    if f.notice_period_days > 60:
        concerns.append(f"a {f.notice_period_days}-day notice period, well above the JD's sub-30-day preference")

    if f.engagement_activity_recency < 0.3:
        concerns.append("has not been active on the platform recently")

    if f.is_closed_source_no_validation:
        concerns.append(
            f"despite {f.years_of_experience:g} years of experience, no visible external validation "
            "(open-source, publications, talks) or active GitHub"
        )

    return concerns


def generate_reasoning(sc: "ScoredCandidate") -> str:
    f = sc.features
    context = f"{f.years_of_experience:g} years as {f.current_title} at {f.current_company}"

    positives = _positive_facts(sc)
    concerns = _concern_facts(sc)

    if positives:
        sentence_1 = f"{context}, with {positives[0]}"
        if len(positives) > 1:
            sentence_1 += f" and {positives[1]}"
        sentence_1 += "."
    else:
        # No qualifying positive fact at all -- say so plainly rather than
        # reusing one of the concern sentences below, which would repeat
        # the same point twice ("no evidence..." / "limited evidence...").
        sentence_1 = f"{context}; no strong direct evidence of retrieval/ranking work in the profile."
        concerns = [c for c in concerns if "retrieval/ranking/vector-DB" not in c]

    if concerns:
        sentence_2 = concerns[0][0].upper() + concerns[0][1:] + "."
    else:
        sentence_2 = ""

    return (sentence_1 + " " + sentence_2).strip()
