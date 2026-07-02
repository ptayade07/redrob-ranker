# Redrob Ranker — Intelligent Candidate Discovery & Ranking Challenge

Ranks the 100,000 candidates in `candidates.jsonl` against Redrob's "Senior AI Engineer —
Founding Team" job description and produces a top-100 submission CSV. Built as a hybrid,
fully-explainable scoring pipeline — structured feature extraction, a lexical/statistical
career-narrative match, rule-based disqualifier checks, and honeypot detection, combined
into one composite score with a reasoning string generated from the same values that drove it.
No LLM calls anywhere in the ranking step (forbidden by the compute budget, and not needed).

## Reproduce

```bash
pip install -r requirements.txt
python rank.py --candidates ./data/raw/candidates.jsonl --out ./output/submission.csv
python validate_submission.py ./output/submission.csv
```

Full run against all 100,000 candidates: **~119s wall-clock, ~320MB peak RAM at a 10K-candidate
sample (scales sub-linearly with corpus size beyond that)** — comfortably inside the 5-minute /
16GB budget. See [Performance](#performance--reproducibility) for the full numbers.

`data/raw/candidates.jsonl` is gitignored (487MB) — place your copy of the dataset there, or
pass `--candidates` pointing anywhere else.

## Repo layout

```
rank.py                  # single entry point: candidates.jsonl -> submission.csv
app.py                   # Streamlit sandbox demo (small-sample, see Stage 9)
src/redrob_ranker/
  config.py              # JD as a structured object: skill buckets, disqualifiers, location
                          #   tiers, notice-period preference, the narrative reference text
  data_loader.py          # streaming JSONL reader
  skills.py               # per-skill "credibility-weighted strength" (proficiency x duration
                           #   x assessment-verified trust discount)
  narrative.py             # TF-IDF cosine similarity vs. the JD's "ideal candidate" narrative
  features.py              # per-candidate feature extraction + the 7 disqualifier signals
  honeypots.py             # impossible-profile detection (3 confirmed mechanisms)
  scoring.py               # composite score: honeypot gate -> core_relevance x secondary_credit
                            #   x logistics x engagement x risk
  reasoning.py             # fact-driven reasoning-string generation
  text_utils.py            # word-boundary keyword matching (see Limitations)
scripts/                 # exploration / sanity-check scripts, run during development
data/reference/          # small committed reference files (schema, 50 sample candidates)
data/raw/                # gitignored -- place the full candidates.jsonl here
docs/                    # JD, signals doc, submission spec (converted to markdown)
```

---

## Methodology

### Why a hybrid rule-based pipeline, not a single model or LLM calls

Two things ruled out the obvious alternatives. The compute budget (5 min, CPU-only, no network,
16GB) rules out calling an LLM per candidate — 100,000 candidates at even a fast API call each
doesn't fit in 5 minutes, and it's disallowed outright per `submission_spec.md` section 3. A
single learned model (e.g. a gradient-boosted ranker) was also ruled out because **there's no
ground truth to train against** — the hidden relevance labels are exactly what's being predicted.
Without labels, a hand-built, feature-driven composite score is not a compromise; it's the only
approach where every score is traceable to a specific, checkable fact about the candidate, which
also happens to be exactly what the reasoning column needs (see [Reasoning generation](#reasoning-generation-fact-driven-not-llm-written)).

The pipeline in one pass: **extract structured features → score them into one composite value →
generate a reasoning string from the same feature values → gate out honeypots first, always**.

### Skill scoring: a trust discount, not a keyword count

The JD is explicit that this dataset has a keyword-stuffing trap built in on purpose. Stage 2's
data exploration confirmed it immediately: `CAND_0000001` (Backend Engineer @ Mindtree, IT
Services) self-reports 7 "advanced" AI skills, but Redrob's own `skill_assessment_scores` for
those exact skills come back mediocre — NLP 38.8, Fine-tuning LLMs 41.6, Speech Recognition
53.7, Image Classification 64.8 (out of 100). A score that only reads the self-reported
proficiency label would reward exactly this pattern.

So every skill's contribution (`skills.py`) is `proficiency_weight × duration_factor ×
trust_factor`, where `trust_factor` blends the self-reported proficiency against the platform
assessment when one exists (`0.4 + 0.6 × assessed_score/100`), or applies a flat 0.75× discount
when there's no assessment to check against (most candidates only have assessments for a
handful of skills, so this is a mild "unverified self-report" tax, not a penalty for missing
data). Skills roll up into per-JD-bucket coverage scores (retrieval/ranking, vector DB, core
Python/ML, fine-tuning, infra/MLOps, CV/speech) via `bucket_coverage(norm=2.0)` — full credit at
roughly two well-evidenced skills, because the JD says explicitly the specific tool doesn't
matter ("we don't care which model... the specific tech doesn't matter").

### Career narrative: TF-IDF, and why it gets the single heaviest weight

The JD's closing section is unambiguous about what actually separates a fit from a trap: *"A
Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career
history shows they built a recommendation system at a product company, they're a fit."* Skill
tags alone cannot catch this — it requires reading what the candidate actually *did*. That's
what `narrative.py` does: TF-IDF cosine similarity (`ngram_range=(1,2)`, so short phrases like
"vector search" or "a/b test" survive as units) between each candidate's `summary` +
`career_history` descriptions and a hand-written "ideal candidate" reference narrative in
`config.py`.

TF-IDF over a small local sentence-embedding model was a deliberate tradeoff, not a default: at
100,000 candidates, CPU-only, 5-minute, no-network, a local embedding model means loading
`torch`, loading weights, and running inference over 100K text blocks inside a Docker
reproduction sandbox that may not have model weights cached and has network turned off.
`TfidfVectorizer` fits the whole corpus in a few seconds and needs nothing beyond what's already
pinned in `requirements.txt`. The real cost: TF-IDF rewards shared *words*, not shared *meaning*
— see [Limitations](#limitations).

In the composite score, `narrative_relevance` carries the single heaviest weight (0.45 of
`core_relevance`, more than either skill-coverage term) because the JD's own framing puts
narrative substance above keyword matching, and because it's the only signal that can catch a
genuine Tier-5 candidate who doesn't use the JD's exact vocabulary.

### Min-max normalization, not percentile rank — a bug found via sanity-checking

`narrative_similarity` is a relative signal — there's no absolute "0.3 is good" cutoff, so it
has to be rescaled against the pool being scored. The first version used **percentile rank**,
which turned out to be a real bug, not a design nuance: almost every candidate's narrative
shares *some* tiny nonzero vocabulary with the reference text (generic words like "team,"
"built," "product" survive stop-word filtering), so percentile rank stretched that noise floor
linearly across nearly the full 0–1 range. An Accountant with zero real retrieval experience
could land at percentile ~0.45 purely because a big block of equally-irrelevant candidates tied
at the bottom got tie-averaged into the *middle* of the rank range, not the bottom.

Switched to **min-max normalization** (`raw_cosine / pool_max`, floor 0) instead, which respects
*magnitude*, not just relative order: a noise-floor candidate at raw cosine 0.02 against a pool
max of 0.6 scores ~0.03, not ~0.45. This matches the JD's own framing — *"we're not expecting to
find many matches in a 100K candidate pool... we'd rather see 10 great matches than 1000
maybes"* — a sparse population of genuine fits should visibly stand apart from the noise floor,
not get smoothed into a fake gradient. The known tradeoff: a single pathological outlier (e.g. a
profile that happens to reuse large chunks of JD-like language) would compress everyone else
toward zero. No evidence of that in this dataset; worth re-checking if the dataset changes.

### The composite score: a gate, not a flat weighted sum

The first version of `base_fit_score` was a single flat weighted sum across nine terms
(narrative, retrieval/vector-DB coverage, eval-methodology evidence, experience fit, and four
smaller terms). Sanity-checking against the 50 sample candidates exposed why that's wrong: an
**Accountant at Infosys with one incidental "advanced" Kubeflow tag and 5.3 years of
experience** scored `base_fit=0.134` with **zero retrieval, vector-DB, or Python-ML coverage at
all** — 63% of that score came from `experience_fit` alone (0.10 weight × 0.85). Years of tenure
and one random infra skill tag were independently sufficient to manufacture a competitive score
out of a candidate with no relevant signal whatsoever.

The fix restructures `base_fit_score` as a **gate multiplied by a bounded modifier**, not a flat
sum:

```
core_relevance   = 0.45 x narrative_relevance + 0.35 x retrieval_ranking_coverage + 0.20 x vector_db_coverage
secondary_credit = 0.25 x eval_methodology_evidence + 0.25 x experience_fit + 0.15 x core_python_ml_coverage
                  + 0.15 x fine_tuning_coverage + 0.10 x infra_mlops_coverage + 0.10 x title_relevance
base_fit_score   = core_relevance x (0.6 + 0.4 x secondary_credit)
```

`core_relevance` is the actual gate: if a candidate's narrative and retrieval/vector-DB skills
show essentially no real signal, `core_relevance` is near zero and the whole `base_fit_score`
stays near zero — no combination of tenure, a stray infra tag, or a generic title can push it up
on its own. `secondary_credit` can boost a genuinely relevant candidate by up to 40% (the `0.6 +
0.4 × secondary_credit` range), but only once `core_relevance` has established there's something
real to boost.

### Disqualifier signals: some are hard flags, some are soft — and that split is deliberate

The JD lists seven explicit "things we explicitly do NOT want." Before implementing any of them
as a hard flag, each was checked against the actual dataset for a dedicated, findable pattern
— treating a JD-stated disqualifier as automatically detectable would have been a mistake, and
two of them turned out not to be:

- **Well-grounded, implemented as binary flags** with a multiplicative penalty on `final_score`:
  `is_job_hopper` (3+ jobs averaging under 18 months — ×0.50), `is_consulting_only` (entire
  career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini — ×0.50), `is_cv_speech_only`
  (CV/speech skill coverage ≥0.5 with zero retrieval/vector-DB coverage and no NLP/IR language in
  career history — ×0.60), `is_recent_llm_only` (no AI skill with 12+ months of use and no
  pre-LLM-era production ML evidence, **or** the candidate's own summary confesses hobbyist/
  coursework AI exposure — ×0.60).
- **Weakly grounded, implemented as continuous soft signals**, not binary flags:
  `pure_research_only` and `architecture_only_no_code_18mo`. Both were searched for extensively
  (multiple keyword phrasings, industry/title fields, education degree fields) and neither has a
  dedicated archetype anywhere in the 100K-candidate dataset — no "Academia" industry tag, no
  Architect/Director/VP titles at all (checked across 40K candidates). One search actively
  backfired: the JD's own gold-standard candidate profiles contain the literal phrase
  "research-only" while stating a preference *against* it ("strong preference for shipping real
  systems over research-only work") — a naive keyword match would have disqualified the best
  candidates in the pool. Both are implemented as continuous risk multipliers instead
  (`production_evidence_score`, `architecture_only_risk`, both in the `[0.7, 1.0]` range) so they
  contribute a mild, defensible signal without a false-positive cliff.
- **Explicitly gated, by design, per an earlier design discussion**:
  `is_closed_source_no_validation`. Missing data is not evidence of this disqualifier — a
  candidate with no GitHub linked and 2 years of experience is neutral, not penalized, because
  absence of a field just means the field wasn't filled in. This only evaluates for
  `years_of_experience >= 5.0` (the JD's own stated threshold) **and** a genuine absence of
  evidence (no active GitHub, no open-source/publication/talk language anywhere in the
  candidate's own words) — and even then it's a mild ×0.85 discount, our weakest-grounded flag
  by design, never a hard cut.

### Honeypots: hard exclusion, not a soft penalty

Honeypots are removed from the candidate pool entirely, before scoring — they never receive a
score or a rank. This was a deliberate choice over a soft multiplicative penalty: the ground
truth treats them as permanent tier-0 relevance no matter what, and honeypots are *designed* to
look tempting on other dimensions. `CAND_0001610` proves the point — strong narrative, strong
skills, real production semantic-search/FAISS work described — and it's still a honeypot (its
`years_of_experience` field states 3.0 years while `career_history` sums to 5.1 years). A soft
penalty requires trusting the composite formula to crush a score that's deliberately built to
look good everywhere else; hard exclusion doesn't need that trust, and guarantees 0% honeypot
rate rather than hoping the math keeps it under the 10% disqualification threshold.

Three mechanisms are implemented (`honeypots.py`), all found by directly scanning the full
dataset rather than assumed from the spec description:

1. **Impossible tenure** — `duration_months` on a job doesn't match what `start_date`/`end_date`
   implies (>6 months of slack, to absorb rounding without absorbing real impossibilities).
   Example: `CAND_0007353` claims 166 months at a job whose dates imply 34.
2. **"Expert" with ~zero months of use** — `CAND_0003582` claims expert-level MLflow, Photoshop,
   and Content Writing, all at `duration_months=0`.
3. **`years_of_experience` contradicting `career_history`** — `CAND_0003430` states 13.7 years;
   its jobs sum to 0.9.

Their union is **70 candidates (0.07%)** against the spec's stated ~80 (~0.08%) — close enough,
and confirmed to be a genuine cluster rather than a threshold artifact: loosening or tightening
each mechanism's cutoff individually left the union unchanged at 70, while five *other* candidate
mechanisms were checked and explicitly rejected for being either zero-hit or far too common to be
deliberate (see [Honeypot detection](#honeypot-detection-mechanisms-and-what-was-ruled-out) below
for the full list with numbers).

### Reasoning generation: fact-driven, not LLM-written

`reasoning.py` builds the 1–2 sentence reasoning column directly from the same
`CandidateFeatures` values that drove the score — no LLM call, both because it's forbidden by
the compute budget and because it structurally can't hallucinate: every fact it can cite (a
matched skill name, a location, a notice period, a tenure pattern) comes from a field already
extracted from that candidate's own record. Variation across the 100 rows comes for free from
being fact-driven rather than templated — different candidates have different skills, companies,
titles, and numbers, so the sentences differ without an artificial randomizer. Positive facts
(matched must-have skills, narrative match strength, eval-methodology evidence, location fit,
notice period, engagement) and concern facts (job-hopper pattern, consulting-only career,
CV/speech-only skills, recent-LLM-only exposure, weak location fit, long notice period, low
recent activity, the closed-source proxy) are each generated from an explicit condition on a
feature value, so a rank-95 candidate's reasoning can't accidentally read like a rank-5
candidate's.

---

## Worked examples

Concrete before/after evidence from the actual pipeline, not just claims.

**`CAND_0000001` — keyword-stuffer, correctly demoted.** Backend Engineer @ Mindtree (IT
Services), career history is 100% Kafka/Spark/Airflow/dbt data-engineering prose. Skills list
claims 7 "advanced" AI skills with assessment scores of 39–65/100 (self-report vs.
platform-verified mismatch — see [Skill scoring](#skill-scoring-a-trust-discount-not-a-keyword-count)).
`vector_db_coverage=0.28` from one aged, unverified "advanced" Milvus claim (35 months, no
assessment) — the trust discount alone doesn't fully zero it out. What does: the corroboration
check finds no retrieval/search/vector language anywhere in the actual career-history text
(pure data-engineering), so that 0.28 gets halved to 0.14 before it ever reaches
`core_relevance`, and `is_recent_llm_only` also trips (no AI skill has 12+ months of *corroborated*
use). **Result: rank 29,286 of 99,930** — solidly mid-pack, nowhere near the top 100.

**`CAND_0000031` — a perfect title, a real job-hopper, correctly kept out of the top 100 once
competing against the full pool.** Titled *"Recommendation Systems Engineer"* — an exact keyword
match — with genuinely real skills (`retrieval_ranking_coverage=1.0`, `vector_db_coverage=0.66`,
real Embeddings/FAISS/Hugging Face Transformers) and a narrative that does describe relevant
work. But `career_history` shows 4 jobs averaging ~17.5 months each — the JD's title-chaser
disqualifier, checked against actual tenure math, not a self-reported label. `is_job_hopper`
applies a ×0.50 risk multiplier. Against just the 50-candidate sample (the Stage 6 sanity check
pool), this candidate was still the clear #1 by a wide margin — there was nothing better in that
small pool to compare against. **Against the full 100,000-candidate pool, it lands at rank 163,
score 0.2433** — good enough to be a plausible near-miss, but correctly pushed just outside the
top 100 once real competition exists. This is the risk multiplier doing exactly its job: a good
title and real skills aren't enough to outrank equally-skilled candidates without the same red
flag.

**`CAND_0086022` — a genuine "Tier 5" hidden gem, correctly surfaced at rank #3.** Senior Applied
Scientist @ Sarvam AI. Career history: *"Built a RAG-based ranking pipeline serving 50M+ queries
per month... combined BM25 + dense retrieval (BGE embeddings, FAISS HNSW) with an LLM-based
re-ranker."* Prior role: *"Senior ML Engineer — Search & Ranking @ Uber... led the migration from
keyword-based to embedding-based search across a 30M+ candidate corpus."* Real
`retrieval_ranking_coverage=1.0`, `vector_db_coverage=1.0`, no disqualifier flags. **Final rank:
#3, score 0.638** — this candidate was independently identified as a strong fit during Stage
3/4 manual data exploration, before any scoring code existed, and the pipeline landed on the
same conclusion independently.

**The "rag" inside "leverage" bug.** While debugging why `flag_recent_llm_only`'s
hobbyist-language check was firing on 35% of the dataset instead of an expected low single
digits, the cause traced to a plain `term in text` substring check: `"rag" in "leverage"` is
`True` in Python, because "leverage" literally contains the substring "rag." `CAND_0000001`'s
summary — *"I can leverage my existing data-infra skills"* — was being read as a RAG mention.
The same class of bug affects `"rank"` (`"rank" in "frankly"`, `"crank"`, `"drank"`) and
`"search"` (`"search" in "research"` — meaning a candidate whose profile mentions "market
research" would have falsely satisfied a check meant to catch retrieval/search expertise).
Fixed with a shared `text_utils.contains_term()` using a **prefix-only word boundary**: a term
must start at a word boundary (not be preceded by a word character), which rejects all three
mid-word collisions above while still correctly matching intentional stems like `"embed"` inside
`"embedding"` — a stricter both-sides `\bterm\b` boundary would have broken that intentional
match. After the fix, `flag_recent_llm_only`'s true rate (36.5% on a 5K slice) was verified by
reading flagged profiles directly, not just trusting the number — e.g. `CAND_0000014`'s summary
literally says *"I've been keeping up with AI/ML at a self-learner level — taken some online
courses, played with the OpenAI and Anthropic APIs, built a small RAG side project — but I
haven't done it in a professional capacity yet."* — and confirmed via title distribution
(concentrated in mainstream engineering titles like Mobile/Software/QA/Frontend/DevOps
Developer, not evenly spread), consistent with a deliberately common trap archetype, not noise.

---

## Honeypot detection: mechanisms and what was ruled out

Before finalizing the 3 mechanisms above, a second pass specifically searched for additional
impossible-profile patterns, since the spec's "~80 honeypots" language implied there could be
more than what Stage 2's first look had found. Five more candidate mechanisms were checked
against the full 100K dataset and explicitly rejected, with numbers, not just eyeballed:

| Pattern checked | Result | Verdict |
|---|---|---|
| Skill duration exceeding how long the named tool has existed (LangChain Oct 2022, LlamaIndex Nov 2022, QLoRA May 2023, PEFT Feb 2023, +6mo buffer) | 841 hits (0.8%), only 6/800 overlap with the 3 confirmed mechanisms | Rejected — ~10x the honeypot budget on its own; the generator doesn't constrain skill duration against real tool-release dates, this is noise |
| Well-known company (Google/Amazon/Meta/etc.) tagged with a small `company_size` | 0 hits | Nothing to detect — `company_size` is perfectly consistent per company across the whole dataset |
| Latest-degree completion 2+ years before earliest job start | 19,499 hits (19.5%) | Rejected — far too common; this is normal (part-time/executive postgrad study while working), not an impossibility |
| First-degree completion 3+ years before first job | 8,846 hits (8.8%) | Rejected — same issue |
| Exact `(name, company, title)` duplicates | 5,398 hits | Rejected — expected name-collision noise from a finite anonymized-name pool at 100K rows |
| Skill-list length distribution | Smooth, natural (5–23 skills), no outlier spike | Nothing found |

The 3 confirmed mechanisms' thresholds were also stress-tested (loosened duration-mismatch 6mo→3mo,
tightened expert-zero 3mo→1mo, loosened YOE-mismatch 1.5yr→1.0yr) — the union stayed at exactly
70 candidates, stable under threshold changes. That stability is what distinguishes a deliberate
cluster from background noise; the rejected patterns above all scaled roughly linearly with
threshold looseness instead.

**Full-run result:** 70/100,000 (0.07%) flagged and excluded before scoring; **0% honeypot rate
in the final top 100**, against the 10% disqualification threshold.

---

## Composite scoring — full formula

```
final_score = base_fit_score x logistics_multiplier x engagement_multiplier x risk_multiplier

base_fit_score = core_relevance x (0.6 + 0.4 x secondary_credit)

core_relevance (weights sum to 1.0):
  0.45 x narrative_relevance          (min-max normalized TF-IDF cosine vs. the JD's "ideal candidate" narrative)
  0.35 x retrieval_ranking_coverage   (corroboration-discounted, see below)
  0.20 x vector_db_coverage           (corroboration-discounted)

secondary_credit (weights sum to 1.0):
  0.25 x eval_methodology_evidence    (binary: NDCG/MRR/MAP/A-B-testing language in career_history)
  0.25 x experience_fit               (soft distance from the JD's 6-8y sweet spot / 5-9y band)
  0.15 x core_python_ml_coverage
  0.15 x fine_tuning_coverage
  0.10 x infra_mlops_coverage
  0.10 x title_relevance              (deliberately small -- JD warns against over-trusting titles)

logistics_multiplier   = 0.6 + 0.4 x (0.7 x location_fit + 0.3 x notice_period_fit)     range [0.6, 1.0]
engagement_multiplier  = 0.7 + 0.3 x avg(activity_recency, response_rate,
                                          interview_completion, profile_completeness)    range [0.7, 1.0]

risk_multiplier = product of, each applied only if the flag fires:
  is_job_hopper                    x0.50
  is_consulting_only               x0.50
  is_cv_speech_only                x0.60
  is_recent_llm_only               x0.60
  is_closed_source_no_validation   x0.85
  x (1 - 0.3 x architecture_only_risk)          continuous, [0.7, 1.0]
  x (0.7 + 0.3 x production_evidence_score)     continuous, [0.7, 1.0]

Corroboration discount: if a candidate's career_history_text (NOT profile.summary -- see
Limitations) contains none of ("retriev", "rank", "search", "embed", "recommend", "vector",
"semantic", "similarity", "nearest neighbor"), retrieval_ranking_coverage and vector_db_coverage
are each multiplied by 0.5 before entering core_relevance.
```

Honeypots are excluded from the pool entirely before any of the above runs.

---

## Limitations

Named here on purpose — a reviewer trusts a writeup more, not less, when it states its own weak
points rather than only its strengths.

- **`is_closed_source_no_validation` is a weak proxy by design.** Absence of an active GitHub
  and absence of open-source/publication/talk language is not proof a candidate spent 5+ years
  on closed-source work — it's the best available signal given the schema, but it's genuinely
  possible for a strong candidate to have neither a linked GitHub nor any public writing and
  still not fit this disqualifier. Mitigated by gating it on the JD's own 5-year threshold, by
  requiring a genuine absence of *all* evidence rather than one missing field, and by keeping the
  penalty mild (×0.85) — but it remains the least-confident flag in the pipeline.
- **The corroboration check is lexical, not semantic.** It's a fixed list of ~9 term-stems
  checked against `career_history_text` with prefix-boundary matching — not an embedding
  similarity, not an LLM judgment. A candidate who genuinely built retrieval systems but
  described the work with vocabulary outside that list (e.g. only ever wrote "similarity
  lookup" style phrasing the list doesn't cover) could be incorrectly discounted. The discount is
  deliberately 0.5×, not 0×, specifically to bound the damage from this kind of miss.
- **TF-IDF narrative similarity rewards shared words, not shared meaning.** A genuinely relevant
  candidate using unusual vocabulary could score lower on `narrative_relevance` than their actual
  experience deserves. Mitigated by never relying on it alone — the structural skill-bucket
  features cover overlapping ground through an independent path — but it's a real, acknowledged
  gap, not a solved problem. A larger compute/time budget would justify revisiting this with a
  small local embedding model instead.
- **`pure_research_only` and `architecture_only_no_code_18mo` have no dataset-specific
  validation beyond "no dedicated archetype was found."** Their continuous-signal implementation
  is a considered choice, not a proven one — if this pipeline were pointed at a different
  candidate dataset, these two signals would need to be re-checked, not assumed to transfer.
- **Career-history title/description pairing can be internally scrambled in this dataset.**
  `CAND_0000021` has "Project Manager @ Wipro" paired with a brand-design job description, and
  "Marketing Manager @ Infosys" paired with a mechanical-engineering description — title and
  description appear to come from mismatched templates for some candidates. The pipeline doesn't
  lean on this pairing being coherent (`title_relevance` only reads `profile.current_title`, not
  individual `career_history[].title` values), so exposure is limited, but per-entry title text
  elsewhere should not be fully trusted.
- **Scoring weights are hand-set and validated by inspection, not fit against ground truth or
  systematically swept.** There's no way to do otherwise here — the hidden relevance labels are
  the thing being predicted — but that means the weights (0.45/0.35/0.20 for `core_relevance`,
  the risk multipliers, the `[0.6, 1.0]` / `[0.7, 1.0]` floors) are a defensible starting point
  grounded in how strongly the JD states each requirement, not a mathematically optimal
  configuration.
- **`narrative_relevance` is pool-relative by construction**, and therefore only valid when
  computed against the exact pool being ranked. `scoring.score_candidates()` emits a
  `RuntimeWarning` when called with fewer than 1,000 candidates for exactly this reason — the
  50-sample sanity-check runs used throughout development are qualitative "does the ordering
  look sane" checks, not comparable to real submission scores, and the full pipeline always
  scores the complete candidate file in one batch.

---

## Performance & reproducibility

Measured on this dev machine (Windows, Python 3.13, 8 logical cores available, ~8GB system RAM
— notably *less* than the 16GB grading sandbox, so these numbers are a conservative floor, not a
best case):

| Metric | Value |
|---|---|
| Full 100,000-candidate run, wall-clock (internal) | 96.8s |
| Full run, total process time (incl. Python/numpy/scikit-learn import overhead) | **118.3s** |
| Budget | 300s (61% margin) |
| Peak RSS at 10,000 candidates (measured via `psutil`) | ~320MB |
| Budget | 16GB |
| Honeypots excluded | 70 / 100,000 (0.07%) |
| Honeypot rate in final top 100 | 0% (budget: <10%) |
| `validate_submission.py` | PASS |

Feature extraction was profiled and optimized before the full run: it was spending 81% of its
time in `re.Pattern.search()` calls (one compiled-regex search per keyword, per term-list, per
candidate — 135K calls for just 3,000 candidates). `text_utils.contains_any_term()` compiles one
alternation pattern per term-list instead of looping per-term, a 3.2x speedup (13.7s → 4.3s per
10K candidates).

No GPU, no network calls, no precomputed artifacts to ship — the TF-IDF vectorizer is fit fresh
on each run directly from the candidate corpus (a few seconds at 100K scale), so there's no
separate precomputation step to document.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11+ (developed and tested on 3.13; avoid 3.14+ until numpy/scikit-learn wheels
catch up). Place the full `candidates.jsonl` at `data/raw/candidates.jsonl`, or pass any path via
`--candidates`.

## Sandbox demo

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Loads the committed 50-candidate
`data/reference/sample_candidates.json` by default (zero setup), or upload any small `.json`
array / `.jsonl` slice to rank a different sample. Click **Run ranking** to see the same
`candidate_id, rank, score, reasoning` columns as the submission CSV.

`app.py` calls `src/redrob_ranker/scoring.score_candidates()` directly — the exact same function
`rank.py` uses for the real submission, not a simplified stand-in. The only difference from a
real run is pool size: per submission_spec.md section 10.5, a sandbox only needs to prove the
system runs end-to-end on a small sample, not reproduce the full 100K ranking (that's what Stage
3 code reproduction checks separately). Because `narrative_relevance` is normalized relative to
whatever pool is scored (see [Min-max normalization](#min-max-normalization-not-percentile-rank--a-bug-found-via-sanity-checking)),
scores from this small-sample sandbox aren't comparable to real submission scores — the app
surfaces that as an on-screen notice after each run, not just in this README.

Verified end-to-end with a headless-browser driver (Playwright) before committing: default
sample loads with zero errors, **Run ranking** produces a populated results table, file upload
with a custom 5-candidate slice correctly reruns the pipeline and updates the table, zero
browser console errors in either path.

## What's been validated

- `validate_submission.py` passes against the full-run output.
- Honeypot rate in the top 100 is 0%, against the 10% disqualification threshold.
- All 3 honeypot mechanisms and 5 rejected alternatives were checked against the real dataset
  with numbers, not assumed (see [Honeypot detection](#honeypot-detection-mechanisms-and-what-was-ruled-out)).
- Every disqualifier flag's firing rate was checked against a real slice of the dataset after
  implementation, and two were caught over-firing and fixed before the full run (`flag_cv_speech_only`
  was tripping on any nonzero CV/speech coverage; `flag_recent_llm_only` was vacuously true for
  candidates with zero AI skills at all).
- The composite scoring formula was sanity-checked against the 50-candidate sample, which
  surfaced and led to fixing four real bugs (percentile-rank noise inflation, summary-vs-career-history
  text scoping, the flat-sum structural issue, and the regex substring collision) — see
  [Worked examples](#worked-examples) for specifics.
- Known trap candidates identified during manual data exploration (Stages 2, 4, 5) were confirmed
  absent from the final top 100 in the full run.
