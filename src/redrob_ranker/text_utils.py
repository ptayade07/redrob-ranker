"""
Shared word-boundary text matching.

Plain `term in text` substring checks silently false-positive on short
terms: "rag" matches inside "leverage"/"average"/"storage", "rank" matches
inside "frankly"/"crank"/"drank", "search" matches inside "re-search".
Found while debugging why flag_recent_llm_only's hobbyist-language check
was firing on 35% of the dataset instead of the expected low single
digits -- CAND_0000001's "I can leverage my existing data-infra skills"
was being read as a "rag" (retrieval-augmented generation) mention. Every
keyword-list match in this codebase goes through this helper now, not
just the one that got caught.

Deliberately PREFIX-only boundary (term must start at a word boundary --
not preceded by a word character), no suffix boundary requirement. This
does two things at once: (1) rejects the mid-word collisions above, since
in "leverage" the "rag" substring is preceded by "e", a word character,
so the prefix check fails; (2) still matches intentional word-stems like
"embed" inside "embedding"/"embeddings" or "retriev" inside "retrieval",
which a full \\bterm\\b (both-sides) boundary would break, since there's
no boundary between "embed" and the "ding"/"s" that follows it.

Performance note: profiling the Stage 6 pipeline on a 10K-candidate slice
showed re.Pattern.search() as 81% of feature-extraction time -- one
separate compiled-regex search per term, per term-list, per candidate
(135K searches for just 3000 candidates). contains_any_term() compiles
ONE alternation pattern per term-list (cached) and does a single search
instead of looping python-side over N patterns.
"""

from __future__ import annotations

import re
from functools import lru_cache

# All term lists in this codebase happen to start with an alphanumeric
# character (checked: skill names, JD vocabulary, hand-written marker
# phrases). That's what lets contains_any_term compile one shared-prefix
# alternation instead of a per-term boundary. If a term list ever needs a
# leading non-alnum term (e.g. "@mention"), _combined_pattern_for will
# raise rather than silently produce a wrong pattern.


@lru_cache(maxsize=256)
def _pattern_for(term: str) -> re.Pattern:
    prefix = r"(?<!\w)" if term[0].isalnum() else ""
    return re.compile(prefix + re.escape(term))


@lru_cache(maxsize=64)
def _combined_pattern_for(terms: tuple[str, ...]) -> re.Pattern:
    if not all(t[0].isalnum() for t in terms):
        raise ValueError(
            "_combined_pattern_for assumes every term starts with an "
            "alphanumeric character so one shared prefix-boundary can "
            "apply to the whole alternation; got a term that doesn't "
            f"(terms={terms!r}). Use contains_term() per-term instead."
        )
    alternation = "|".join(re.escape(t) for t in terms)
    return re.compile(r"(?<!\w)(?:" + alternation + ")")


def contains_term(text: str, term: str) -> bool:
    """Prefix-word-boundary match -- see module docstring for why prefix-only."""
    return _pattern_for(term).search(text) is not None


def contains_any_term(text: str, terms) -> bool:
    return _combined_pattern_for(tuple(terms)).search(text) is not None


def count_distinct_terms(text: str, terms) -> int:
    """How many distinct terms from `terms` appear in `text`, each counted
    once regardless of repetition -- for signals that care about breadth
    of evidence (e.g. production_evidence_score), not just presence.
    """
    pattern = _combined_pattern_for(tuple(terms))
    return len({m.group() for m in pattern.finditer(text)})
