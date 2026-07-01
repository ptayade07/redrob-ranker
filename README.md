# Redrob Ranker — Intelligent Candidate Discovery & Ranking Challenge

Ranks the 100,000 candidates in `candidates.jsonl` against the Redrob "Senior AI Engineer —
Founding Team" job description and produces the top-100 submission CSV.

## TODO

- [ ] Setup instructions
- [ ] Exact reproduce command
- [ ] Methodology writeup (feature extraction, narrative similarity, honeypot detection,
      scoring, reasoning generation) — with the *why* behind each design choice
- [ ] Known limitations / things we'd do with more time
- [ ] Sandbox demo instructions (`app.py`)

## Repo layout (draft, will finalize in Stage 8)

```
rank.py                    # single entry point: candidates.jsonl -> submission.csv
app.py                     # Streamlit sandbox demo
src/redrob_ranker/         # pipeline source
scripts/                   # exploration / one-off analysis scripts
data/reference/            # small committed reference files (schema, sample candidates)
data/raw/                  # gitignored — place the full candidates.jsonl here
docs/                      # JD, signals doc, submission spec (converted to markdown)
```

## Reproduce (placeholder — finalized in Stage 7)

```bash
python rank.py --candidates ./data/raw/candidates.jsonl --out ./output/submission.csv
```
