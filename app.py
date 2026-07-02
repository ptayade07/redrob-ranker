"""
Minimal Streamlit sandbox demo.

Satisfies submission_spec.md section 10.5 (sandbox/demo requirement): accept
a small candidate sample (<=100 candidates), run the ranking system
end-to-end, produce a ranked table, complete within the compute budget. It
does not need to handle the full 100K pool -- that's what Stage 3 code
reproduction checks separately.

Calls the exact same src/redrob_ranker/scoring.score_candidates() used by
rank.py -- no separate or simplified scoring logic here. The point of a
sandbox is proving the real system reproduces, not a stand-in that could
silently drift from the real pipeline.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.scoring import score_candidates  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE_PATH = REPO_ROOT / "data" / "reference" / "sample_candidates.json"
MAX_DISPLAY_ROWS = 100
SUBMISSION_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def load_candidates(raw_text: str, filename: str) -> list[dict]:
    """Accepts a JSON array (sample_candidates.json's format) or JSONL
    (candidates.jsonl's format), regardless of which extension is used --
    a small uploaded slice cut from the full file could be either.
    """
    if filename.lower().endswith(".jsonl") or not raw_text.strip().startswith("["):
        return [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    return json.loads(raw_text)


st.set_page_config(page_title="Redrob Ranker Sandbox")
st.title("Redrob Ranker — Sandbox")
st.caption(
    "Runs the exact same ranking pipeline as `rank.py` "
    "(`src/redrob_ranker/scoring.score_candidates`) against a small candidate sample."
)

uploaded = st.file_uploader(
    "Candidate sample (.json array or .jsonl) -- optional, defaults to the "
    "committed 50-candidate sample below",
    type=["json", "jsonl"],
)

if uploaded is not None:
    candidates = load_candidates(uploaded.getvalue().decode("utf-8"), uploaded.name)
    source_label = uploaded.name
else:
    candidates = json.loads(DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8"))
    source_label = "data/reference/sample_candidates.json (default)"

st.write(f"Loaded **{len(candidates)}** candidates from `{source_label}`.")

if st.button("Run ranking"):
    with st.spinner("Scoring..."):
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scored, honeypots_excluded = score_candidates(candidates)
        elapsed = time.time() - t0

    st.success(
        f"Scored {len(scored)} candidates ({len(honeypots_excluded)} honeypots "
        f"excluded) in {elapsed:.2f}s."
    )

    for w in caught:
        # score_candidates() warns (plain UserWarning) whenever the pool is
        # small -- exactly the case for this sandbox by design (see
        # submission_spec.md 10.5: "does not need to handle the full 100K
        # pool"). Surface it so the score isn't mistaken for a real
        # submission score.
        if issubclass(w.category, UserWarning):
            st.info(str(w.message))

    top = scored[:MAX_DISPLAY_ROWS]
    df = pd.DataFrame(
        [
            {"candidate_id": s.candidate_id, "rank": i, "score": round(s.final_score, 6),
             "reasoning": s.reasoning}
            for i, s in enumerate(top, start=1)
        ],
        columns=SUBMISSION_COLUMNS,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    if honeypots_excluded:
        with st.expander(f"{len(honeypots_excluded)} honeypot(s) excluded from ranking"):
            for cid, reasons in honeypots_excluded:
                st.write(f"**{cid}**: {reasons}")
