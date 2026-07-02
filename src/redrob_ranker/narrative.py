"""
TF-IDF cosine-similarity scorer for career-narrative text.

Why TF-IDF and not a sentence-embedding model: at 100,000 candidates, CPU-only,
5-minute wall clock, no network (submission_spec.md section 3), a local
sentence-transformer checkpoint is riskier than it looks -- even a small
model means loading torch, loading weights, and running inference over
100K text blocks inside a Docker reproduction sandbox that may not have the
model weights cached and has network turned off. TfidfVectorizer + cosine
similarity fits the whole 100K corpus in a few seconds, needs nothing
beyond what's already in requirements.txt, and has zero download step.

This is a real tradeoff, not a free lunch: TF-IDF rewards shared *words*,
not shared *meaning*, so a genuinely relevant profile written with unusual
vocabulary could score lower than it deserves. We mitigate two ways: (1)
ngram_range=(1,2) so short domain phrases like "vector search" or "a/b
test" match as units instead of losing their meaning when split into
unigrams, and (2) this score is one signal among several in the composite
(Stage 6) -- it's never used alone to catch or reject a "Tier 5" candidate,
the structural skill-bucket features in skills.py cover the same ground
from a different angle.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def candidate_narrative_text(candidate: dict) -> str:
    """Concatenates the free-text fields that describe what a candidate has
    actually done: summary, headline, and every career_history description.
    Deliberately excludes the skills list -- that's scored structurally in
    skills.py, and mixing it into the narrative text would let keyword-
    stuffed skill tags inflate the similarity score too, defeating the
    point of having two independent signals.
    """
    profile = candidate["profile"]
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    parts.extend(h.get("description", "") for h in candidate["career_history"])
    return " ".join(p for p in parts if p)


class NarrativeSimilarityScorer:
    def __init__(self, reference_text: str):
        self._reference_text = reference_text
        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        self._reference_vector = None

    def fit(self, corpus_texts: list[str]) -> "NarrativeSimilarityScorer":
        # Fit IDF weights on the candidate corpus + the reference narrative
        # together, so the reference text's own vocabulary participates in
        # the same term space instead of being silently dropped for being
        # out-of-vocabulary.
        self._vectorizer.fit(corpus_texts + [self._reference_text])
        self._reference_vector = self._vectorizer.transform([self._reference_text])
        return self

    def score_many(self, texts: list[str]) -> np.ndarray:
        if self._reference_vector is None:
            raise RuntimeError("call .fit(corpus_texts) before scoring")
        matrix = self._vectorizer.transform(texts)
        return cosine_similarity(matrix, self._reference_vector).ravel()

    def score_one(self, text: str) -> float:
        return float(self.score_many([text])[0])
