"""
Structured representation of job_description.md.

This module exists so that every scoring rule in the pipeline traces back to
one place, and that place quotes or paraphrases a specific line of the JD.
Nothing in here is tuned against the ground truth (we don't have it) --
these are read-the-JD-literally values, not fitted hyperparameters.

Skill name buckets below were built from the actual controlled vocabulary
observed in candidates.jsonl (scripts/explore_data.py can regenerate the
frequency table), not guessed -- e.g. "Fine-tuning LLMs" and "Learning to
Rank" are the literal strings the generator uses, so alias handling is
minimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Skill vocabulary
#
# JD section: "The skills inventory (please read carefully)". The JD is
# explicit that the *specific* tool doesn't matter ("we don't care which
# model", "the specific tech doesn't matter; the operational experience
# does") -- so these are buckets of interchangeable tools, not a single
# required skill each.
# ---------------------------------------------------------------------------

# "Production experience with embeddings-based retrieval systems ... and
# production experience with vector databases or hybrid search
# infrastructure." These two JD bullets are the strongest must-haves, so
# they get the largest weight in Stage 4's skill-fit feature.
RETRIEVAL_RANKING_SKILLS: tuple[str, ...] = (
    "Sentence Transformers", "Embeddings", "RAG", "Vector Search",
    "Semantic Search", "Information Retrieval", "Hugging Face Transformers",
    "Recommendation Systems", "Learning to Rank", "BM25",
)

VECTOR_DB_SKILLS: tuple[str, ...] = (
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
    "Elasticsearch", "FAISS", "pgvector", "LlamaIndex", "Haystack",
)

# "Strong Python." The JD only names one hard language requirement.
CORE_PYTHON_ML_SKILLS: tuple[str, ...] = (
    "Python", "PyTorch", "TensorFlow", "scikit-learn", "Machine Learning",
    "Deep Learning",
)

# "Things we'd like you to have but won't reject you for: LLM fine-tuning
# experience (LoRA, QLoRA, PEFT)."
FINE_TUNING_SKILLS: tuple[str, ...] = (
    "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT", "LLMs",
)

# "Prior exposure to HR-tech, recruiting tech, or marketplace products" and
# "distributed systems or large-scale inference optimization" -- operational
# maturity signals, not core to the role but a plus.
INFRA_MLOPS_SKILLS: tuple[str, ...] = (
    "MLOps", "MLflow", "Kubeflow", "BentoML", "Kubernetes", "Docker",
    "gRPC", "Microservices", "Weights & Biases",
)

# "People whose primary expertise is computer vision, speech, or robotics
# without significant NLP/IR exposure. We respect your work but you'd be
# re-learning fundamentals here." Not a keyword blacklist -- see
# src/redrob_ranker/features.py, which only treats this as a negative when
# a candidate's *entire* AI skill set falls in this bucket with zero
# overlap against RETRIEVAL_RANKING_SKILLS / VECTOR_DB_SKILLS.
CV_SPEECH_ROBOTICS_SKILLS: tuple[str, ...] = (
    "Computer Vision", "OpenCV", "CNN", "Image Classification", "GANs",
    "TTS", "ASR", "Speech Recognition", "YOLO", "Object Detection",
    "Diffusion Models",
)

# "Framework enthusiasts. If your GitHub is full of LangChain tutorials ...
# that's fine but it's not what we need." These tags alone (without any
# RETRIEVAL_RANKING_SKILLS / VECTOR_DB_SKILLS backing them) are the
# "keyword stuffer" signature the JD's closing section calls out directly.
FRAMEWORK_ONLY_MARKERS: tuple[str, ...] = ("LangChain", "Prompt Engineering")

# Neutral ML/data context -- not scored as a JD requirement on its own, but
# used to distinguish "has general ML background" from "has zero ML
# background" when a candidate is missing everything above.
GENERAL_DATA_SKILLS: tuple[str, ...] = (
    "Feature Engineering", "Time Series", "Forecasting",
    "Statistical Modeling", "Data Science", "Reinforcement Learning", "NLP",
)

# Evaluation-methodology vocabulary. These are almost never literal skill
# tags in this dataset (checked: not present in the top-150 skill
# frequency table) -- the JD's ask ("hands-on experience designing
# evaluation frameworks ... NDCG, MRR, MAP, offline-to-online correlation,
# A/B test interpretation") shows up in career_history *prose* instead.
# Stage 4 searches description text for these rather than the skills list.
EVAL_METHODOLOGY_TERMS: tuple[str, ...] = (
    "ndcg", "mrr", "map@", "mean average precision", "a/b test", "ab test",
    "offline-to-online", "offline to online", "precision@", "recall@",
    "click-through rate", "ctr", "learning-to-rank", "learning to rank",
)


# ---------------------------------------------------------------------------
# Location tiers
#
# JD: "Location: Pune/Noida-preferred but flexible... Candidates in
# Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply... Open to relocation
# candidates from Tier-1 Indian cities... Outside India: case-by-case, but
# we don't sponsor work visas."
#
# City lists below were cross-checked against the actual `location` values
# in candidates.jsonl (18 Indian cities appear, each ~1200-1300 candidates
# in a 30K-row sample) so every India-based candidate lands in a defined
# tier instead of falling through to "unknown".
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocationTier:
    tier: int
    label: str
    cities: tuple[str, ...]
    weight: float  # 1.0 = ideal, decays for less-preferred tiers


LOCATION_TIERS: tuple[LocationTier, ...] = (
    # JD names these two by name as the actual office locations.
    LocationTier(1, "JD-preferred (Pune/Noida)", ("Pune", "Noida"), 1.00),
    # JD names these explicitly as "welcome to apply".
    LocationTier(2, "JD-named acceptable", ("Hyderabad", "Mumbai", "Delhi", "Gurgaon"), 0.85),
    # Other Tier-1 Indian cities per "open to relocation candidates from
    # Tier-1 Indian cities" -- not named individually but covered by that
    # sentence.
    LocationTier(3, "Other Tier-1 India", ("Bangalore", "Chennai", "Kolkata"), 0.65),
    # Remaining India locations seen in the dataset: still no visa/relocation
    # blocker, just not called out in the JD, so a smaller penalty than
    # tier 4/5.
    LocationTier(4, "Other India", (
        "Indore", "Ahmedabad", "Bhubaneswar", "Jaipur", "Trivandrum",
        "Vizag", "Chandigarh", "Coimbatore", "Kochi",
    ), 0.45),
)

# "Outside India: case-by-case, but we don't sponsor work visas." Not a hard
# zero (JD explicitly leaves the door open) but the steepest discount --
# and willingness to relocate matters a lot more here than for India-based
# candidates (see features.py).
NON_INDIA_BASE_WEIGHT = 0.15
NON_INDIA_RELOCATE_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# Notice period
#
# JD: "We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day
# notice candidates are still in scope but the bar gets higher."
# ---------------------------------------------------------------------------

IDEAL_NOTICE_PERIOD_DAYS = 30       # full credit at/under this
STEEP_PENALTY_NOTICE_PERIOD_DAYS = 90  # beyond this, treat as a real drag on score


# ---------------------------------------------------------------------------
# Experience band
#
# JD: "'5-9 years' ... is a range, not a requirement ... we'll seriously
# consider candidates outside the band if other signals are strong." Kept
# as a soft center-of-mass, not a filter -- see features.py.
# ---------------------------------------------------------------------------

EXPERIENCE_BAND_YEARS = (5.0, 9.0)
EXPERIENCE_SWEET_SPOT_YEARS = (6.0, 8.0)  # "the 'ideal candidate' ... 6-8 years total"


# ---------------------------------------------------------------------------
# Explicit disqualifiers
#
# Each entry names the JD sentence it encodes and which module implements
# the actual check, so the mapping from "JD says X" to "code does Y" is
# traceable without re-reading the whole JD.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Disqualifier:
    key: str
    jd_quote: str
    implemented_in: str


DISQUALIFIERS: tuple[Disqualifier, ...] = (
    Disqualifier(
        "pure_research_only",
        "If you've spent your career in pure research environments (academic "
        "labs, research-only roles) without any production deployment -- we "
        "will not move forward.",
        "features.py: flag_pure_research_only() -- CONTINUOUS 0..1 risk score, "
        "not a binary flag. Stage 4 data exploration searched career_history "
        "text for academic/research-only language (PhD, postdoc, thesis, "
        "'never shipped', 'academic lab', etc.) and found zero hits in the "
        "full 100K-candidate dataset -- this JD disqualifier has no dedicated "
        "keyword/title archetype here (unlike consulting_only or "
        "title_chaser, which do). Naively matching the phrase 'research-only' "
        "even backfires: it appears in the JD's own gold-standard "
        "candidates' summaries as a stated preference AGAINST research-only "
        "work ('strong preference for shipping real systems over "
        "research-only work'). So this is implemented as the inverse of a "
        "continuous production-evidence score (count of shipping/deployment "
        "language across career_history), which the narrative-similarity "
        "feature (Stage 6) already reinforces, rather than a fragile hard "
        "trigger that risks false-positiving on strong candidates.",
    ),
    Disqualifier(
        "recent_llm_only_no_pre_llm_production",
        "If your 'AI experience' consists primarily of recent (under 12 "
        "months) projects using LangChain to call OpenAI -- we will probably "
        "not move forward, unless you can demonstrate substantial pre-LLM-era "
        "ML production experience.",
        "features.py: flag_recent_llm_only()",
    ),
    Disqualifier(
        "architecture_only_no_code_18mo",
        "If you are a senior engineer who hasn't written production code in "
        "the last 18 months because you've moved into 'architecture' or "
        "'tech lead' roles -- we will probably not move forward.",
        "features.py: flag_architecture_only() -- CONTINUOUS 0..1 risk score. "
        "The dataset's generated title vocabulary has no 'Architect', "
        "'Engineering Manager', 'Director', or 'VP' titles at all (checked "
        "across 40K candidates), so there's no dedicated trap to hard-match "
        "against. Implemented as a forward-looking structural check (current "
        "role title matches an architecture/lead marker AND tenure in that "
        "role >= 18 months) that will correctly return 0 for this dataset "
        "but generalizes if the title vocabulary ever includes those roles.",
    ),
    Disqualifier(
        "title_chaser",
        "Title-chasers. If your career trajectory shows you optimizing for "
        "'Senior' -> 'Staff' -> 'Principal' titles by switching companies "
        "every 1.5 years, we're not a fit.",
        "features.py: flag_job_hopper()",
    ),
    Disqualifier(
        "consulting_only",
        "People who have only worked at consulting firms (TCS, Infosys, "
        "Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire "
        "career ... If you're currently at one of these companies but have "
        "prior product-company experience, that's fine.",
        "features.py: flag_consulting_only()",
    ),
    Disqualifier(
        "cv_speech_robotics_only",
        "People whose primary expertise is computer vision, speech, or "
        "robotics without significant NLP/IR exposure.",
        "features.py: flag_cv_speech_only()",
    ),
    Disqualifier(
        "closed_source_no_validation",
        "People whose work has been entirely on closed-source proprietary "
        "systems for 5+ years without external validation (papers, talks, "
        "open-source).",
        "features.py: flag_closed_source_no_validation() "
        "(proxied via github_activity_score / certifications -- the "
        "dataset has no papers/talks field)",
    ),
)

CONSULTING_FIRMS: tuple[str, ...] = (
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
)


# ---------------------------------------------------------------------------
# "Ideal candidate" narrative
#
# JD: "How to read between the lines ... The 'ideal candidate' we're
# imagining is roughly: ...". Written in first-person career-history prose
# (not JD boilerplate) on purpose -- Stage 6 cosine-similarity-matches this
# against each candidate's own career_history text, and TF-IDF/embedding
# similarity is far more reliable when the reference text is stylistically
# close to what it's being compared against, not a bullet-point job ad.
# This is also the mechanism that's supposed to catch "Tier 5" candidates
# who never say "RAG" or "Pinecone" but whose career narrative is the real
# thing.
# ---------------------------------------------------------------------------

IDEAL_CANDIDATE_NARRATIVE = """
Built and owned a production ranking, search, or recommendation system end
to end at a product company, serving real users at meaningful scale. Worked
on the retrieval layer -- embeddings, nearest-neighbor search, hybrid
retrieval combining sparse and dense signals -- and owned it through
production concerns like embedding drift, index refresh, and retrieval
quality regressions, not just a notebook prototype. Designed and ran
offline evaluation for ranking quality using metrics like NDCG, MRR, or MAP,
and connected offline gains to online A/B test results. Comfortable with
both the pre-LLM-era ranking stack (learning-to-rank models, feature
engineering, gradient-boosted rankers) and modern LLM-based re-ranking or
retrieval-augmented generation, and can explain the tradeoffs between them.
Writes production Python. Shipped iteratively rather than only publishing
research, and has some form of external validation of the work -- open
source contributions, technical writing, or conference talks -- rather than
five-plus years entirely behind a closed door.
""".strip()


@dataclass(frozen=True)
class JDRequirements:
    retrieval_ranking_skills: tuple[str, ...] = RETRIEVAL_RANKING_SKILLS
    vector_db_skills: tuple[str, ...] = VECTOR_DB_SKILLS
    core_python_ml_skills: tuple[str, ...] = CORE_PYTHON_ML_SKILLS
    fine_tuning_skills: tuple[str, ...] = FINE_TUNING_SKILLS
    infra_mlops_skills: tuple[str, ...] = INFRA_MLOPS_SKILLS
    cv_speech_robotics_skills: tuple[str, ...] = CV_SPEECH_ROBOTICS_SKILLS
    framework_only_markers: tuple[str, ...] = FRAMEWORK_ONLY_MARKERS
    general_data_skills: tuple[str, ...] = GENERAL_DATA_SKILLS
    eval_methodology_terms: tuple[str, ...] = EVAL_METHODOLOGY_TERMS
    location_tiers: tuple[LocationTier, ...] = LOCATION_TIERS
    non_india_base_weight: float = NON_INDIA_BASE_WEIGHT
    non_india_relocate_weight: float = NON_INDIA_RELOCATE_WEIGHT
    ideal_notice_period_days: int = IDEAL_NOTICE_PERIOD_DAYS
    steep_penalty_notice_period_days: int = STEEP_PENALTY_NOTICE_PERIOD_DAYS
    experience_band_years: tuple[float, float] = EXPERIENCE_BAND_YEARS
    experience_sweet_spot_years: tuple[float, float] = EXPERIENCE_SWEET_SPOT_YEARS
    disqualifiers: tuple[Disqualifier, ...] = DISQUALIFIERS
    consulting_firms: tuple[str, ...] = CONSULTING_FIRMS
    ideal_candidate_narrative: str = IDEAL_CANDIDATE_NARRATIVE


JD = JDRequirements()
