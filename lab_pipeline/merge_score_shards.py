"""Merge per-file AF3 score shard CSVs into one windows_scored.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    root = Path(args.shards_dir)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise SystemExit(f"No shard CSVs in {root}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # Drop exact duplicate windows if a shard was re-run
    key = [c for c in ["file_id", "start_second"] if c in df.columns]
    if key:
        df = df.drop_duplicates(subset=key, keep="last")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Merged {len(files)} shards, {len(df)} windows -> {out}")


if __name__ == "__main__":
    main()
