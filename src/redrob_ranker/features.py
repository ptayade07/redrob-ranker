"""
Per-candidate feature extraction.

Turns one raw candidate dict into a CandidateFeatures record: a flat set of
0..1 fit scores, boolean/continuous disqualifier signals, and raw context
fields carried forward for reasoning generation (Stage 6). Every function
here maps back to a specific line in config.py / job_description.md --
see the Disqualifier.jd_quote fields in config.py for the exact wording
each flag_* function is checking.

Narrative similarity (TF-IDF cosine vs. the JD's "ideal candidate"
narrative) is NOT computed here -- it needs the whole candidate corpus
fitted at once (see narrative.py), so it's attached to CandidateFeatures
in a separate batch step during the pipeline run (Stage 6/7), not per
candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import skills
from .config import JD, JDRequirements
from .text_utils import contains_any_term, contains_term, count_distinct_terms

# Dataset generation reference point. last_active_date / signup_date values
# in this dataset cluster right up through mid-2026 (checked against the
# sample), so "now" for recency math is the dataset's own clock, not
# whatever day the pipeline happens to run on.
DATASET_NOW = date(2026, 7, 1)

# --- title relevance ---------------------------------------------------
# Keyword tiers, not an exhaustive whitelist -- new/unseen titles fall
# through to a low default rather than erroring. This is intentionally a
# coarse, low-weight signal: the JD explicitly warns against over-trusting
# titles/keywords, so title_relevance_score contributes modestly to the
# composite (Stage 6) alongside the heavier skill-bucket and narrative
# signals, rather than gating on it.
_STRONG_TITLE_KEYWORDS = (
    "recommendation", "ranking", "search", "retrieval", "nlp", "ai engineer",
    "ai research", "applied scientist", "machine learning engineer", "ml engineer",
)
_MODERATE_TITLE_KEYWORDS = (
    "data scientist", "data engineer", "software engineer", "backend engineer",
    "analytics engineer", "computer vision",
)
_WEAK_TITLE_KEYWORDS = (
    "devops", "qa engineer", "frontend engineer", "mobile developer",
    "cloud engineer", "full stack", "java developer", ".net developer",
)


def title_relevance_score(title: str) -> float:
    t = title.lower()
    if any(k in t for k in _STRONG_TITLE_KEYWORDS):
        return 1.0
    if any(k in t for k in _MODERATE_TITLE_KEYWORDS):
        return 0.55
    if any(k in t for k in _WEAK_TITLE_KEYWORDS):
        return 0.3
    return 0.1  # e.g. Marketing Manager, HR Manager, Sales Executive, Accountant


# --- experience band -----------------------------------------------------

def experience_band_fit(years: float, jd: JDRequirements = JD) -> float:
    lo, hi = jd.experience_band_years
    slo, shi = jd.experience_sweet_spot_years
    if slo <= years <= shi:
        return 1.0
    if lo <= years <= hi:
        return 0.85
    gap = (lo - years) if years < lo else (years - hi)
    # JD: "we'll seriously consider candidates outside the band if other
    # signals are strong" -- gentle decay with a floor, not a cliff.
    return max(0.3, 0.85 - 0.1 * gap)


# --- location --------------------------------------------------------------

def location_fit(candidate: dict, jd: JDRequirements = JD) -> float:
    profile = candidate["profile"]
    if profile["country"] != "India":
        willing = candidate["redrob_signals"].get("willing_to_relocate", False)
        return jd.non_india_relocate_weight if willing else jd.non_india_base_weight

    city = profile["location"].split(",")[0].strip()
    for tier in jd.location_tiers:
        if city in tier.cities:
            return tier.weight
    return 0.45  # unlisted Indian city -- same default as the "other India" tier


# --- notice period -----------------------------------------------------

def notice_period_fit(notice_days: int, jd: JDRequirements = JD) -> float:
    ideal, steep = jd.ideal_notice_period_days, jd.steep_penalty_notice_period_days
    if notice_days <= ideal:
        return 1.0
    if notice_days >= steep:
        return 0.35
    span = steep - ideal
    return 1.0 - 0.65 * (notice_days - ideal) / span


# --- disqualifier / risk signals ----------------------------------------

def flag_job_hopper(career_history: list[dict], min_jobs: int = 3, max_avg_tenure_months: float = 18) -> bool:
    """JD: 'switching companies every 1.5 years' -- checked directly against
    career_history tenure math, not a self-reported label. See CAND_0000031
    in scripts/explore_data.py output: title is a perfect keyword match
    ("Recommendation Systems Engineer") but tenure pattern is exactly this.
    """
    if len(career_history) < min_jobs:
        return False
    avg_tenure = sum(h["duration_months"] for h in career_history) / len(career_history)
    return avg_tenure < max_avg_tenure_months


def flag_consulting_only(career_history: list[dict], jd: JDRequirements = JD) -> bool:
    """JD: consulting-only career is a disqualifier, UNLESS there's prior
    product-company experience -- which this check honors automatically:
    if any single company falls outside the consulting set, this is False.
    """
    companies = {h["company"].lower() for h in career_history}
    return bool(companies) and companies.issubset(set(jd.consulting_firms))


_NLP_IR_OVERRIDE_TERMS = (
    "nlp", "natural language", "information retrieval", "search relevance",
    "ranking", "recommendation", "retrieval",
)


# Below this, a candidate has at most one weakly-evidenced CV/speech skill
# tag -- not enough to call it their "primary expertise," which is the
# JD's actual bar ("primary expertise is computer vision, speech, or
# robotics"). 0.5 roughly corresponds to one well-evidenced skill match
# given bucket_coverage's norm=2.0 for this bucket.
_CV_SPEECH_PRIMARY_EXPERTISE_THRESHOLD = 0.5


def flag_cv_speech_only(
    cv_speech_coverage: float,
    retrieval_ranking_coverage: float,
    vector_db_coverage: float,
    career_text: str,
) -> bool:
    """JD: 'primary expertise is computer vision, speech, or robotics
    without significant NLP/IR exposure.' Only trips when CV/speech
    coverage is substantial enough to call it a "primary expertise" AND
    retrieval/vector-DB skill coverage is zero AND no NLP/IR language
    appears anywhere in the career narrative -- a candidate who lists
    Computer Vision alongside real IR work, or who only has one incidental
    CV skill tag, is exactly the kind of profile the JD says NOT to penalize.
    """
    if cv_speech_coverage < _CV_SPEECH_PRIMARY_EXPERTISE_THRESHOLD:
        return False
    if retrieval_ranking_coverage > 0 or vector_db_coverage > 0:
        return False
    text = career_text.lower()
    return not contains_any_term(text, _NLP_IR_OVERRIDE_TERMS)


_PRE_LLM_ML_TERMS = (
    "machine learning", "model", "pipeline", "ranking", "recommendation",
    "nlp", "data science", "ml",
)
# ChatGPT's public launch -- rough dividing line between "pre-LLM-era
# production ML" and "recent LLM-wrapper work" per the JD's own framing.
_LLM_ERA_START = "2022-11-01"


def _any_pre_llm_production_ml(career_history: list[dict]) -> bool:
    for h in career_history:
        if h["start_date"] < _LLM_ERA_START and contains_any_term(h["description"].lower(), _PRE_LLM_ML_TERMS):
            return True
    return False


_HOBBYIST_AI_MARKERS = (
    "online course", "online courses", "side project", "side projects",
    "personal project", "personal projects", "for fun", "in my spare time",
    "self-directed", "self-taught", "experimenting with", "exploring how",
)
_HOBBYIST_AI_TOPIC_TERMS = (
    "rag", "llm", "langchain", "vector database", "genai", "openai", "gpt",
    "prompt engineering", "fine-tun",
)


def _summary_confesses_hobbyist_ai(summary: str) -> bool:
    """Catches candidates whose own profile summary frames their AI
    exposure as coursework/side-projects rather than production work --
    e.g. CAND_0000021 (Stage 6 conversation): 'I've been taking online
    courses on RAG and vector databases, experimenting with LangChain and
    the OpenAI API for side projects.' That candidate's skills list
    self-reports 18 months of 'Embeddings' with zero backing in
    skill_assessment_scores -- an unverified duration claim contradicted
    by the candidate's own narrative. This check trusts the narrative over
    the duration field when they disagree.
    """
    text = summary.lower()
    return contains_any_term(text, _HOBBYIST_AI_MARKERS) and contains_any_term(text, _HOBBYIST_AI_TOPIC_TERMS)


def flag_recent_llm_only(candidate: dict, jd: JDRequirements = JD) -> bool:
    """JD: 'AI experience consists primarily of recent (under 12 months)
    projects using LangChain to call OpenAI ... unless you can demonstrate
    substantial pre-LLM-era ML production experience.'

    Two independent checks, either one is sufficient:
    1. The candidate's own summary frames their AI exposure as hobbyist/
       coursework (see _summary_confesses_hobbyist_ai) -- this overrides
       self-reported skill durations, which aren't independently verified.
    2. Structural fallback for candidates who don't narrate it explicitly:
       the candidate HAS at least one AI-flavored skill (otherwise this
       disqualifier doesn't apply -- "no AI experience" is a different,
       separately-captured failure mode), none of those skills has 12+
       months of claimed use, AND no career_history entry predating the
       LLM era describes ML/data work.
    """
    if _summary_confesses_hobbyist_ai(candidate["profile"].get("summary", "")):
        return True

    ai_bucket = (
        jd.retrieval_ranking_skills + jd.vector_db_skills + jd.fine_tuning_skills
        + jd.framework_only_markers + jd.cv_speech_robotics_skills + jd.general_data_skills
    )
    ai_bucket_lower = {s.lower() for s in ai_bucket}
    ai_durations = [s["duration_months"] for s in candidate["skills"] if s["name"].lower() in ai_bucket_lower]
    if not ai_durations:
        return False
    if max(ai_durations) >= 12:
        return False
    return not _any_pre_llm_production_ml(candidate["career_history"])


_PRODUCTION_EVIDENCE_TERMS = (
    "deployed", "production", "shipped", "serving", "served", "live",
    "launched", "scale", "real users", "a/b test", "rolled out", "in prod",
)


def production_evidence_score(career_text: str) -> float:
    """Continuous 0..1 proxy for 'not pure research only' -- see config.py's
    pure_research_only Disqualifier note for why this is continuous rather
    than a hard keyword-match flag (naive matching backfires on this JD:
    the gold-standard candidate profiles literally contain the phrase
    'research-only' while describing their preference AGAINST it).
    """
    text = career_text.lower()
    # Distinct-term presence count, not raw substring occurrence count --
    # a candidate repeating "production" five times shouldn't outscore one
    # with diverse evidence across three different terms.
    hits = count_distinct_terms(text, _PRODUCTION_EVIDENCE_TERMS)
    return min(1.0, hits / 3)  # 3+ distinct terms present = full credit


_ARCHITECTURE_TITLE_MARKERS = ("architect", "engineering manager", "director", "vp", "head of", "principal")


def architecture_only_risk(career_history: list[dict]) -> float:
    """Continuous 0..1 -- see config.py's architecture_only_no_code_18mo
    Disqualifier note: this dataset's title vocabulary has no Architect/
    Director/VP titles at all, so this will be 0.0 for every candidate here,
    but the check is real and would activate if that ever changes.
    """
    if not career_history:
        return 0.0
    current = next((h for h in career_history if h["is_current"]), career_history[0])
    if not contains_any_term(current["title"].lower(), _ARCHITECTURE_TITLE_MARKERS):
        return 0.0
    tenure = current["duration_months"]
    return min(1.0, tenure / 18) if tenure >= 18 else 0.3


_EXTERNAL_VALIDATION_TERMS = (
    "open source", "open-source", "published", "publication", "paper",
    "arxiv", "blog", "talk", "conference", "presented", "speaker",
)


def flag_closed_source_no_validation(candidate: dict) -> bool:
    """JD: closed-source-only for 5+ years with no external validation.

    Confirmed design (see conversation): missing data is NOT evidence of
    this disqualifier. A candidate with no GitHub linked and under 5 years
    of experience is neutral, not penalized -- absence of a field just
    means the field wasn't filled in. This only fires when the JD's own
    "5+ years" threshold is met AND there is a complete absence of any
    external-validation evidence (active GitHub OR publication/talk/
    open-source language in their own words).
    """
    if candidate["profile"]["years_of_experience"] < 5.0:
        return False
    github_score = candidate["redrob_signals"].get("github_activity_score", -1)
    if github_score is not None and github_score > 0:
        return False
    text = (candidate["profile"].get("summary", "") + " "
            + " ".join(h["description"] for h in candidate["career_history"])).lower()
    if contains_any_term(text, _EXTERNAL_VALIDATION_TERMS):
        return False
    return True


def eval_methodology_evidence(career_text: str, jd: JDRequirements = JD) -> bool:
    """JD: 'hands-on experience designing evaluation frameworks ... NDCG,
    MRR, MAP, offline-to-online correlation, A/B test interpretation.'
    These terms are not skill tags in this dataset (checked: absent from
    the top-150 skill frequency table) -- they appear in career_history
    prose instead, so this is a text search, not a skills-list lookup.
    """
    text = career_text.lower()
    return contains_any_term(text, jd.eval_methodology_terms)


# --- behavioral signals --------------------------------------------------

def activity_recency_score(last_active_date: str, full_credit_days: int = 30, zero_credit_days: int = 180) -> float:
    """JD: 'a perfect-on-paper candidate who hasn't logged in for 6 months
    ... is, for hiring purposes, not actually available.' zero_credit_days=180
    (~6 months) is a direct match to the JD's own example.
    """
    y, m, d = map(int, last_active_date.split("-"))
    days = (DATASET_NOW - date(y, m, d)).days
    if days <= full_credit_days:
        return 1.0
    if days >= zero_credit_days:
        return 0.1
    span = zero_credit_days - full_credit_days
    return 1.0 - 0.9 * (days - full_credit_days) / span


# --- the feature record ---------------------------------------------------

@dataclass
class CandidateFeatures:
    candidate_id: str

    # skill fit
    retrieval_ranking_coverage: float
    vector_db_coverage: float
    core_python_ml_coverage: float
    fine_tuning_coverage: float
    infra_mlops_coverage: float
    cv_speech_coverage: float
    matched_must_have_skills: list[str]
    matched_nice_to_have_skills: list[str]
    matched_cv_speech_skills: list[str]
    eval_methodology_evidence: bool

    # career-history stats, always computed (not just when is_job_hopper),
    # so reasoning generation can cite exact numbers instead of vague
    # language -- e.g. "4 roles averaging 15 months" rather than "changes
    # jobs frequently."
    n_jobs: int
    avg_tenure_months: float
    consulting_companies: list[str]  # non-empty only when is_consulting_only

    # title / seniority / experience
    title_relevance: float
    experience_fit: float

    # location / logistics
    location_fit: float
    notice_period_fit: float

    # disqualifier / risk signals
    is_job_hopper: bool
    is_consulting_only: bool
    is_cv_speech_only: bool
    is_recent_llm_only: bool
    is_closed_source_no_validation: bool
    production_evidence_score: float
    architecture_only_risk: float

    # behavioral signals (raw component scores, combined into a multiplier in Stage 6)
    engagement_activity_recency: float
    engagement_response_rate: float
    engagement_interview_completion: float
    engagement_offer_acceptance: float | None  # None if -1 sentinel: no prior offers
    profile_completeness: float
    open_to_work: bool

    # filled in after batch TF-IDF fit (Stage 6) -- default 0.0 until then.
    # narrative_relevance is narrative_similarity min-max scaled by the
    # pool's own max value (0=no shared vocabulary at all, 1=the best
    # narrative match in the pool) -- see scoring.py's module docstring.
    # This is pool-size- and pool-composition-dependent by design, so it
    # must always be computed against the full candidate pool being
    # ranked, never a subset, or the number means something different
    # than it looks like it means.
    narrative_similarity: float = 0.0
    narrative_relevance: float = 0.0

    # raw context carried forward for reasoning generation (Stage 6)
    current_title: str = ""
    current_company: str = ""
    current_industry: str = ""
    years_of_experience: float = 0.0
    location: str = ""
    country: str = ""
    notice_period_days: int = 0

    narrative_text: str = field(default="", repr=False)
    # career_history descriptions only, no summary/headline -- used for the
    # skill/narrative corroboration check in scoring.py. Deliberately
    # narrower than narrative_text: job descriptions describe verified past
    # work, while profile.summary is self-promotional/aspirational and can
    # contain forward-looking or hobbyist language ("I've been taking
    # online courses on RAG...", "looking to grow into...") that mentions
    # the right words without describing real production experience. See
    # CAND_0000021 in the Stage 6 conversation: their summary contains the
    # word "vector" while explicitly describing side-project/course work,
    # which would have (incorrectly) satisfied a corroboration check that
    # included the summary.
    career_history_text: str = field(default="", repr=False)


def extract_features(candidate: dict, jd: JDRequirements = JD) -> CandidateFeatures:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    career_history = candidate["career_history"]
    skill_list = candidate["skills"]
    assessment_scores = signals.get("skill_assessment_scores", {})

    career_history_text = " ".join(h.get("description", "") for h in career_history)
    narrative_text = " ".join(
        [profile.get("headline", ""), profile.get("summary", ""), career_history_text]
    )

    retrieval_cov, retrieval_matched = skills.bucket_coverage(
        skill_list, assessment_scores, jd.retrieval_ranking_skills, norm=2.0)
    vector_db_cov, vector_db_matched = skills.bucket_coverage(
        skill_list, assessment_scores, jd.vector_db_skills, norm=2.0)
    python_ml_cov, python_ml_matched = skills.bucket_coverage(
        skill_list, assessment_scores, jd.core_python_ml_skills, norm=1.5)
    fine_tuning_cov, fine_tuning_matched = skills.bucket_coverage(
        skill_list, assessment_scores, jd.fine_tuning_skills, norm=1.5)
    infra_cov, infra_matched = skills.bucket_coverage(
        skill_list, assessment_scores, jd.infra_mlops_skills, norm=1.5)
    cv_speech_cov, cv_speech_matched = skills.bucket_coverage(
        skill_list, assessment_scores, jd.cv_speech_robotics_skills, norm=2.0)

    must_have_names = sorted({s.name for s in retrieval_matched + vector_db_matched + python_ml_matched})
    nice_to_have_names = sorted({s.name for s in fine_tuning_matched + infra_matched})
    cv_speech_names = sorted({s.name for s in cv_speech_matched})

    offer_rate = signals.get("offer_acceptance_rate", -1)

    n_jobs = len(career_history)
    avg_tenure_months = sum(h["duration_months"] for h in career_history) / n_jobs if n_jobs else 0.0
    is_consulting = flag_consulting_only(career_history, jd)
    consulting_companies = sorted({h["company"] for h in career_history}) if is_consulting else []

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        retrieval_ranking_coverage=retrieval_cov,
        vector_db_coverage=vector_db_cov,
        core_python_ml_coverage=python_ml_cov,
        fine_tuning_coverage=fine_tuning_cov,
        infra_mlops_coverage=infra_cov,
        cv_speech_coverage=cv_speech_cov,
        matched_must_have_skills=must_have_names,
        matched_nice_to_have_skills=nice_to_have_names,
        matched_cv_speech_skills=cv_speech_names,
        eval_methodology_evidence=eval_methodology_evidence(career_history_text, jd),
        n_jobs=n_jobs,
        avg_tenure_months=avg_tenure_months,
        consulting_companies=consulting_companies,

        title_relevance=title_relevance_score(profile["current_title"]),
        experience_fit=experience_band_fit(profile["years_of_experience"], jd),

        location_fit=location_fit(candidate, jd),
        notice_period_fit=notice_period_fit(signals["notice_period_days"], jd),

        is_job_hopper=flag_job_hopper(career_history),
        is_consulting_only=is_consulting,
        is_cv_speech_only=flag_cv_speech_only(cv_speech_cov, retrieval_cov, vector_db_cov, career_history_text),
        is_recent_llm_only=flag_recent_llm_only(candidate, jd),
        is_closed_source_no_validation=flag_closed_source_no_validation(candidate),
        production_evidence_score=production_evidence_score(career_history_text),
        architecture_only_risk=architecture_only_risk(career_history),

        engagement_activity_recency=activity_recency_score(signals["last_active_date"]),
        engagement_response_rate=signals["recruiter_response_rate"],
        engagement_interview_completion=signals["interview_completion_rate"],
        engagement_offer_acceptance=None if offer_rate is None or offer_rate < 0 else offer_rate,
        profile_completeness=signals["profile_completeness_score"] / 100,
        open_to_work=signals["open_to_work_flag"],

        current_title=profile["current_title"],
        current_company=profile["current_company"],
        current_industry=profile["current_industry"],
        years_of_experience=profile["years_of_experience"],
        location=profile["location"],
        country=profile["country"],
        notice_period_days=signals["notice_period_days"],

        narrative_text=narrative_text,
        career_history_text=career_history_text,
    )
