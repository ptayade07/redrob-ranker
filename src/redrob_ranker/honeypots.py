"""
Honeypot / impossible-profile detection.

submission_spec.md section 7: "~80 honeypot candidates with subtly
impossible profiles ... forced to relevance tier 0 in the ground truth.
If your submission ranks honeypots in the top 10 ... submissions with
honeypot rate > 10% in top 100 are disqualified."

The three mechanisms below were found by directly scanning the full
100K-candidate dataset (see scripts/explore_data.py for the original
Stage 2 findings, and the Stage 5 conversation for a second pass that
specifically searched for additional mechanisms before this module was
written). Their union is 70 candidates and is STABLE under threshold
changes (loosening/tightening each cutoff individually still nets ~70) --
that stability is what distinguishes them from noise.

Five other candidate mechanisms were checked and explicitly rejected for
being either zero-hit (nothing to detect: e.g. well-known companies
always carry a consistent company_size, so there's no inconsistency to
find there) or far too common to be a deliberate trap (e.g. "earliest
degree completes 3+ years after the first job starts" hits 8.8% of the
dataset -- a normal executive-education/part-time-postgrad pattern, not
an impossibility; "skill duration exceeds how long the named tool has
publicly existed" hits 0.8% of the dataset with almost no overlap against
the confirmed mechanisms, i.e. ~10x the honeypot budget on its own --
the generator simply doesn't constrain skill duration_months against
real-world tool release dates, so this is generator noise, not a signal).
Using either of those as a honeypot flag would have falsely tanked
thousands of otherwise-legitimate candidates, so they're deliberately
left out. See the Stage 5 conversation for the full list of checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# "Now" for date math -- same reference point as features.py, since
# last_active_date / career_history dates in this dataset cluster right up
# through mid-2026.
DATASET_NOW = "2026-07-01"

# A career_history entry's stated duration_months vs. what start_date/
# end_date actually implies. >6 months of slack absorbs rounding in how
# the generator computed "months" (e.g. 30 vs 31-day months, inclusive/
# exclusive day counting) without absorbing genuine impossibilities --
# Stage 2's example, CAND_0007353, claims 166 stated months against a
# calculated 34, an order of magnitude off, nowhere near this threshold.
DURATION_MISMATCH_THRESHOLD_MONTHS = 6

# "Expert" proficiency claimed with next-to-no actual time using the
# skill. Stage 2's example, CAND_0003582, claims expert MLflow/Photoshop/
# Content Writing all at duration_months=0.
EXPERT_ZERO_DURATION_THRESHOLD_MONTHS = 3

# profile.years_of_experience vs. sum(career_history duration_months) in
# years. Stage 2's example, CAND_0003430, states 13.7 years while its
# career_history sums to 0.9 years.
YOE_MISMATCH_THRESHOLD_YEARS = 1.5


def _months_between(start: str, end: str | None) -> int:
    y1, m1, _ = map(int, start.split("-"))
    end = end or DATASET_NOW
    y2, m2, _ = map(int, end.split("-"))
    return (y2 - y1) * 12 + (m2 - m1)


@dataclass
class HoneypotResult:
    candidate_id: str
    is_honeypot: bool
    reasons: list[str] = field(default_factory=list)


def check_impossible_tenure(career_history: list[dict]) -> list[str]:
    reasons = []
    for h in career_history:
        calc = _months_between(h["start_date"], h["end_date"])
        stated = h["duration_months"]
        if abs(calc - stated) > DURATION_MISMATCH_THRESHOLD_MONTHS:
            reasons.append(
                f"claims {stated} months at {h['company']} ({h['title']}) but "
                f"{h['start_date']} to {h['end_date'] or 'present'} is only {calc} months"
            )
    return reasons


def check_expert_zero_duration(skills: list[dict]) -> list[str]:
    reasons = []
    for s in skills:
        if s["proficiency"] == "expert" and s["duration_months"] <= EXPERT_ZERO_DURATION_THRESHOLD_MONTHS:
            reasons.append(f"claims expert '{s['name']}' with only {s['duration_months']} months of use")
    return reasons


def check_years_of_experience_mismatch(profile: dict, career_history: list[dict]) -> list[str]:
    total_months = sum(h["duration_months"] for h in career_history)
    calc_years = total_months / 12
    stated_years = profile["years_of_experience"]
    if abs(calc_years - stated_years) > YOE_MISMATCH_THRESHOLD_YEARS:
        return [
            f"profile states {stated_years} years of experience but career_history "
            f"sums to {calc_years:.1f} years"
        ]
    return []


def check_honeypot(candidate: dict) -> HoneypotResult:
    reasons = []
    reasons.extend(check_impossible_tenure(candidate["career_history"]))
    reasons.extend(check_expert_zero_duration(candidate["skills"]))
    reasons.extend(check_years_of_experience_mismatch(candidate["profile"], candidate["career_history"]))
    return HoneypotResult(
        candidate_id=candidate["candidate_id"],
        is_honeypot=bool(reasons),
        reasons=reasons,
    )
