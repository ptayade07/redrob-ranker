"""
Composite scoring: combines a CandidateFeatures record into one final_score
per candidate, gates out honeypots, and attaches a grounded reasoning string.

IMPORTANT -- pool-dependence of narrative_relevance:
narrative_similarity (see narrative.py) is a TF-IDF cosine similarity, which
is only meaningful in relative terms -- there's no absolute "0.3 is good"
cutoff. So it's min-max normalized *within whatever pool is passed to
score_candidates()* (divided by that pool's own max similarity) before
being weighted into the composite. That means a candidate's contribution
from this term is pool-size- and pool-composition-dependent: their
normalized score against 50 sample candidates is a different number than
their normalized score against the full 100,000.

Consequence: score_candidates() must always be called with the FULL
candidate pool being ranked, in one batch, for the real submission --
never a subset. Calling it with a small pool (e.g. the 50-candidate
sample, for a dev sanity-check) is fine for eyeballing whether the
*ordering* looks sane, but the resulting scores are not comparable to,
and must never be reported as, real submission scores. To make that
mistake hard to make silently, score_candidates() emits a RuntimeWarning
whenever the pool is smaller than FULL_POOL_SIZE_WARNING_THRESHOLD.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import features as feat_mod
from .config import JD, JDRequirements
from .honeypots import check_honeypot
from .narrative import NarrativeSimilarityScorer
from .reasoning import generate_reasoning
from .text_utils import contains_any_term

# Below this, pool-relative narrative scoring is statistically thin and
# this is almost certainly a dev sanity-check run, not the real 100K pass.
FULL_POOL_SIZE_WARNING_THRESHOLD = 1_000

# Lexical corroboration check for the retrieval_ranking / vector_db skill
# buckets specifically -- see Stage 6 design conversation. Targets the
# keyword-stuffing trap directly: CAND_0000001 claims Milvus (advanced,
# unverified) but their career narrative is pure data-engineering prose
# with zero retrieval/search/vector language anywhere in it. Checked
# against career_history_text ONLY, not the full narrative (which also
# includes profile.summary) -- CAND_0000021's summary contains the word
# "vector" while describing side-project/coursework AI exposure, which
# would have (incorrectly) satisfied this check against the full text.
# career_history describes verified past work; summary is self-promotional
# and can contain aspirational language about the right words.
CORROBORATION_TERMS = (
    "retriev", "rank", "search", "embed", "recommend", "vector", "semantic",
    "similarity", "nearest neighbor",
)
CORROBORATION_DISCOUNT = 0.5  # not zero -- the lexical check is a heuristic,
                               # not proof of stuffing; see Stage 6 conversation.

# base_fit_score = CORE_RELEVANCE_WEIGHTS-weighted gate x SECONDARY_WEIGHTS-
# weighted modifier -- NOT a single flat weighted sum. See Stage 6
# conversation: a flat sum let "years of experience" and one incidental
# infra skill tag accumulate enough independent credit to outrank
# candidates with real Milvus/FAISS evidence, because a candidate with
# ZERO retrieval/vector-DB signal could still max out experience_fit
# (0.10 weight) and title_relevance's floor plus a random nice-to-have
# skill hit -- e.g. an Accountant with one incidental "advanced" Kubeflow
# tag and 5.3 years of experience scored base_fit=0.134 with literally no
# retrieval/vector-DB/Python-ML coverage at all, 63% of it from
# experience_fit alone. That's backwards: years of tenure and a stray
# infra tag shouldn't manufacture relevance out of nothing.
#
# core_relevance is the actual gate: if a candidate's narrative and
# retrieval/vector-DB skills show ~no real signal, core_relevance is ~0
# and the whole base_fit stays ~0 no matter how strong the secondary
# factors are. narrative_relevance carries the heaviest weight within it
# on purpose -- the JD's closing section explicitly prioritizes
# career-narrative substance over keyword/title matching ("reasoning
# about the gap between what the JD says and what the JD means").
CORE_RELEVANCE_WEIGHTS = {
    "narrative_relevance": 0.45,
    "retrieval_ranking_coverage": 0.35,
    "vector_db_coverage": 0.20,
}
assert abs(sum(CORE_RELEVANCE_WEIGHTS.values()) - 1.0) < 1e-9

# secondary_credit can boost a genuinely relevant candidate (nonzero
# core_relevance) by up to 40% -- see the (0.6 + 0.4 * secondary_credit)
# multiplier below -- but can't create relevance on its own.
SECONDARY_WEIGHTS = {
    "eval_methodology_evidence": 0.25,
    "experience_fit": 0.25,
    "core_python_ml_coverage": 0.15,
    "fine_tuning_coverage": 0.15,
    "infra_mlops_coverage": 0.10,
    "title_relevance": 0.10,
}
assert abs(sum(SECONDARY_WEIGHTS.values()) - 1.0) < 1e-9

# Multiplicative disqualifier penalties. Independent flags compound rather
# than average -- a candidate hitting two explicit JD disqualifiers at once
# should be penalized more than either alone, not less.
RISK_MULTIPLIERS = {
    "is_job_hopper": 0.50,                    # JD: "we're not a fit" -- well-grounded trap
    "is_consulting_only": 0.50,               # explicit disqualifier, well-grounded
    "is_cv_speech_only": 0.60,                # JD's tone is softer ("we respect your work but...")
    "is_recent_llm_only": 0.60,               # JD: "probably not move forward"
    "is_closed_source_no_validation": 0.85,   # weakest-grounded flag by design -- mild only
}


@dataclass
class ScoredCandidate:
    candidate_id: str
    final_score: float
    features: feat_mod.CandidateFeatures
    base_fit_score: float
    core_relevance: float
    secondary_credit: float
    logistics_multiplier: float
    engagement_multiplier: float
    risk_multiplier: float
    narrative_corroborated: bool
    reasoning: str = ""


def _narrative_corroborated(career_history_text: str) -> bool:
    text = career_history_text.lower()
    return contains_any_term(text, CORROBORATION_TERMS)


def _normalize_narrative_similarity(values: np.ndarray) -> np.ndarray:
    """0..1 score, scaled by the pool's own max (min is 0 -- TF-IDF cosine
    similarity of non-negative vectors can't go below that).

    This started as a percentile rank and was changed after the Stage 6
    sample sanity-check surfaced a real bug, not a sample-size artifact:
    percentile rank is a pure order statistic, so it doesn't care how much
    better one value is than another, only their relative position. Almost
    every candidate's narrative has SOME tiny nonzero cosine overlap with
    the reference text (generic words like "team"/"built"/"product"
    survive stop-word filtering), so percentile rank stretched that noise
    floor linearly across nearly the full 0..1 range -- e.g. if 90% of a
    pool sits at or near the minimum value, tie-averaging alone puts that
    whole block around percentile ~0.45, handing a Marketing Manager with
    zero real retrieval/ranking narrative content the same-ish credit as
    a candidate with genuine, if modest, relevant experience.

    Min-max scaling respects magnitude instead of just rank order: a
    noise-floor candidate at raw cosine 0.02 against a pool max of 0.6
    scores ~0.03, not ~0.45. This matches the JD's own framing ("we're
    not expecting to find many matches in a 100K candidate pool ... 10
    great matches, not 1000 maybes") -- a sparse population of genuine
    fits should visibly stand apart from the noise floor, not get smoothed
    into a fake gradient. The tradeoff: a single pathological outlier
    (e.g. a profile that happens to reuse large chunks of JD-like
    language) would compress everyone else toward zero. We found no
    evidence of that in this dataset when checking the sample (the
    observed max, ~0.625 for CAND_0000031, is a real, substantively
    matching profile, not an artifact) -- worth re-checking at full scale
    in Stage 7.
    """
    max_val = float(values.max()) if len(values) else 0.0
    if max_val <= 0:
        return np.zeros(len(values))
    return np.clip(values / max_val, 0.0, 1.0)


def score_candidates(candidates: list[dict], jd: JDRequirements = JD) -> tuple[list[ScoredCandidate], list[tuple[str, list[str]]]]:
    """Returns (scored_candidates sorted best-first, honeypots_excluded).

    honeypots_excluded is [(candidate_id, reasons), ...] for diagnostics --
    these candidates never appear in scored_candidates at all.
    """
    n = len(candidates)
    if n < FULL_POOL_SIZE_WARNING_THRESHOLD:
        warnings.warn(
            f"score_candidates() called with only {n} candidates. "
            "narrative_relevance is min-max normalized WITHIN this pool, "
            "so these scores are only valid for checking relative "
            "ordering within this run -- they are NOT comparable to a full "
            "100,000-candidate run and must never be reported as real "
            "submission scores. Expected for a sample-based sanity check; "
            "should never happen for the real pipeline.",
            stacklevel=2,
        )

    # Gate 1: honeypots never enter the ranking pool.
    survivors = []
    honeypots_excluded = []
    for c in candidates:
        result = check_honeypot(c)
        if result.is_honeypot:
            honeypots_excluded.append((c["candidate_id"], result.reasons))
        else:
            survivors.append(c)

    feats = [feat_mod.extract_features(c, jd) for c in survivors]

    # Narrative similarity + pool-relative normalization, fit once over the
    # whole (honeypot-excluded) pool being scored in THIS call.
    texts = [f.narrative_text for f in feats]
    scorer = NarrativeSimilarityScorer(jd.ideal_candidate_narrative).fit(texts)
    raw_sims = scorer.score_many(texts)
    normalized = _normalize_narrative_similarity(raw_sims)
    for f, sim, rel in zip(feats, raw_sims, normalized):
        f.narrative_similarity = float(sim)
        f.narrative_relevance = float(rel)

    scored = []
    for f in feats:
        corroborated = _narrative_corroborated(f.career_history_text)
        retrieval_eff = f.retrieval_ranking_coverage * (1.0 if corroborated else CORROBORATION_DISCOUNT)
        vecdb_eff = f.vector_db_coverage * (1.0 if corroborated else CORROBORATION_DISCOUNT)

        core_relevance = (
            CORE_RELEVANCE_WEIGHTS["narrative_relevance"] * f.narrative_relevance
            + CORE_RELEVANCE_WEIGHTS["retrieval_ranking_coverage"] * retrieval_eff
            + CORE_RELEVANCE_WEIGHTS["vector_db_coverage"] * vecdb_eff
        )

        secondary_credit = (
            SECONDARY_WEIGHTS["eval_methodology_evidence"] * (1.0 if f.eval_methodology_evidence else 0.0)
            + SECONDARY_WEIGHTS["experience_fit"] * f.experience_fit
            + SECONDARY_WEIGHTS["core_python_ml_coverage"] * f.core_python_ml_coverage
            + SECONDARY_WEIGHTS["fine_tuning_coverage"] * f.fine_tuning_coverage
            + SECONDARY_WEIGHTS["infra_mlops_coverage"] * f.infra_mlops_coverage
            + SECONDARY_WEIGHTS["title_relevance"] * f.title_relevance
        )

        base_fit = core_relevance * (0.6 + 0.4 * secondary_credit)

        # [0.6, 1.0] -- JD treats location as "flexible"/"case-by-case," so
        # logistics can meaningfully hurt a score but never zero it alone.
        logistics_mult = 0.6 + 0.4 * (0.7 * f.location_fit + 0.3 * f.notice_period_fit)

        # [0.7, 1.0] -- JD: "down-weight appropriately," explicitly not "exclude."
        engagement_avg = sum([
            f.engagement_activity_recency, f.engagement_response_rate,
            f.engagement_interview_completion, f.profile_completeness,
        ]) / 4
        engagement_mult = 0.7 + 0.3 * engagement_avg

        risk_mult = 1.0
        for flag_name, penalty in RISK_MULTIPLIERS.items():
            if getattr(f, flag_name):
                risk_mult *= penalty
        risk_mult *= (1 - 0.3 * f.architecture_only_risk)          # [0.7, 1.0]
        risk_mult *= (0.7 + 0.3 * f.production_evidence_score)     # [0.7, 1.0]

        final_score = base_fit * logistics_mult * engagement_mult * risk_mult

        scored.append(ScoredCandidate(
            candidate_id=f.candidate_id,
            final_score=final_score,
            features=f,
            base_fit_score=base_fit,
            core_relevance=core_relevance,
            secondary_credit=secondary_credit,
            logistics_multiplier=logistics_mult,
            engagement_multiplier=engagement_mult,
            risk_multiplier=risk_mult,
            narrative_corroborated=corroborated,
        ))

    for s in scored:
        s.reasoning = generate_reasoning(s)

    # Sort by score desc, tie-break candidate_id asc -- matches
    # validate_submission.py's tie-break rule exactly (section 3).
    scored.sort(key=lambda s: (-s.final_score, s.candidate_id))
    return scored, honeypots_excluded
