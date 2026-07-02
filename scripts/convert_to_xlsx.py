"""Converts output/submission.csv to output/submission.xlsx, same columns
and data, no transformation -- some versions of the brief ask for XLSX
instead of CSV; submission_spec.md's own validator (validate_submission.py)
only reads CSV, so this is an additive format conversion, not a replacement.

Usage:
    python scripts/convert_to_xlsx.py [--csv path] [--xlsx path]
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "output" / "submission.csv"
DEFAULT_XLSX = REPO_ROOT / "output" / "submission.xlsx"


def main():
    parser = argparse.ArgumentParser(description="Convert the submission CSV to XLSX, unchanged data.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.to_excel(args.xlsx, index=False, engine="openpyxl")
    print(f"Wrote {len(df)} rows to {args.xlsx}")


if __name__ == "__main__":
    main()
