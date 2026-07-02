"""
Skill relevance scoring with a proficiency/duration/assessment "trust discount".

Why the discount exists at all: Stage 2's data exploration (scripts/explore_data.py)
found candidates who self-report many skills as "advanced"/"expert" while
Redrob's own skill_assessment_scores for those exact skills are mediocre or
simply absent -- e.g. CAND_0000001 claims 7 "advanced" AI skills with
assessment scores of 39-65/100. The JD explicitly says the dataset has this
trap built in on purpose ("the 'right answer' ... is not 'find candidates
whose skills section contains the most AI keywords'"). A skill score that
only reads the self-reported proficiency label would reward exactly that
pattern, so every skill's contribution here is discounted by (a) how well
the self-report is backed by a platform-verified assessment, when one
exists, and (b) how long the candidate has actually used it.
"""

from __future__ import annotations

from dataclasses import dataclass

PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

# 3 years of continuous use = full duration credit. Chosen because it's
# long enough to separate "used it on one project" from "actually built a
# career around it," short enough that it doesn't unfairly punish someone
# whose retrieval work is genuinely recent.
DURATION_FULL_CREDIT_MONTHS = 36
DURATION_FLOOR = 0.4  # a skill used for 0 months still counts for something -- new hires exist

# Flat discount applied when a skill has no Redrob assessment score to back
# it up. Most candidates only have assessments for a handful of skills (the
# schema's skill_assessment_scores dict is sparse), so this is a mild
# "unverified self-report" tax, not a punishment for missing data.
UNVERIFIED_TRUST = 0.75


def _duration_factor(duration_months: int) -> float:
    capped = min(max(duration_months, 0), DURATION_FULL_CREDIT_MONTHS)
    return DURATION_FLOOR + (1 - DURATION_FLOOR) * (capped / DURATION_FULL_CREDIT_MONTHS)


def _trust_factor(skill_name: str, assessment_scores: dict[str, float]) -> float:
    score = assessment_scores.get(skill_name)
    if score is None:
        return UNVERIFIED_TRUST
    # Blend self-report and platform-verified assessment evenly -- a claimed
    # "expert" (weight 1.0) with an assessed score of 40/100 should land
    # closer to "intermediate" credibility than "expert", not be zeroed out
    # entirely (the assessment could be an off day, not fraud).
    return 0.4 + 0.6 * (score / 100)


@dataclass(frozen=True)
class SkillStrength:
    name: str
    proficiency: str
    strength: float  # credibility-weighted strength of this claimed skill


def score_skill(skill: dict, assessment_scores: dict[str, float]) -> SkillStrength:
    prof_weight = PROFICIENCY_WEIGHT.get(skill["proficiency"], 0.25)
    dur_factor = _duration_factor(skill.get("duration_months", 0))
    trust = _trust_factor(skill["name"], assessment_scores)
    return SkillStrength(skill["name"], skill["proficiency"], prof_weight * dur_factor * trust)


def bucket_coverage(
    skills: list[dict],
    assessment_scores: dict[str, float],
    bucket: tuple[str, ...],
    norm: float = 2.0,
) -> tuple[float, list[SkillStrength]]:
    """0..1 coverage score for a JD skill bucket, plus the matched skills.

    norm=2.0 means ~2 well-evidenced, credible skills in this bucket earns
    full credit -- the JD is explicit that the specific tool doesn't matter
    ("we don't care which model ... the specific tech doesn't matter"), so
    covering a couple of tools deeply is treated as equivalent to covering
    all ten shallowly.
    """
    bucket_lower = {b.lower() for b in bucket}
    matched = [score_skill(s, assessment_scores) for s in skills if s["name"].lower() in bucket_lower]
    total = sum(m.strength for m in matched)
    coverage = min(1.0, total / norm) if norm > 0 else 0.0
    return coverage, matched
