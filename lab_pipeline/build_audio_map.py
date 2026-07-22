"""
Build audio_map.csv from a folder of WAVs (multi-file participants OK).

file_id = filename stem (preserves recording identity / clock times).
participant_id = leading digits of stem when present, else stem.

Usage:
  python -m lab_pipeline.build_audio_map \\
    --audio-dir D:/lab_audio \\
    --out-csv lab_runs/my_run/audio_map.csv \\
    --recursive
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def participant_from_stem(stem: str) -> str:
    m = re.match(r"^(\d+)", stem)
    return m.group(1) if m else stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audio_map.csv for lab pipeline")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    root = Path(args.audio_dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    wavs = sorted(root.rglob("*.wav") if args.recursive else root.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No WAVs in {root}")

    rows = []
    for wav in wavs:
        stem = wav.stem
        rows.append(
            {
                "participant_id": participant_from_stem(stem),
                "file_id": stem,
                "wav_path": str(wav.resolve()),
            }
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows -> {out}")
    print(f"Participants: {pd.DataFrame(rows)['participant_id'].nunique()}")


if __name__ == "__main__":
    main()
