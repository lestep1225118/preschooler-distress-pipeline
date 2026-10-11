"""
Count conversational turns and speech duration from VTC (rttm.csv) output.

Turn definition (Dr. Ravindran):
  Speech by one party followed by speech by the other within TURN_GAP_SEC seconds
  (gap from end of first segment to start of second).

Turn pairs reported:
  - Adult ↔ key child (FEM/MAL ↔ KCHI), with FEM and MAL breakdown
  - Male adult ↔ female adult (MAL ↔ FEM)
  - Key child ↔ other child (KCHI ↔ OCH)

Also reports total speech duration (seconds) per label per 5-minute block:
  FEM, MAL, KCHI, OCH (clip segments to block boundaries).

Usage:
  python -m vtc_pipeline.conversational_turns --rttm path/to/rttm.csv --out-dir out/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

TURN_GAP_SEC = 5.0
BLOCK_SEC = 300  # 5 minutes
SPEECH_LABELS = ("FEM", "MAL", "KCHI", "OCH")

TURN_SPECS: Tuple[Tuple[str, Set[str], Set[str]], ...] = (
    ("adult_kchi", {"FEM", "MAL"}, {"KCHI"}),
    ("mal_fem", {"MAL"}, {"FEM"}),
    ("kchi_och", {"KCHI"}, {"OCH"}),
)


def load_rttm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"uid", "start_time_s", "duration_s", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"rttm CSV missing columns: {missing}")
    df = df.copy()
    df["label"] = df["label"].astype(str)
    df["end_time_s"] = df["start_time_s"] + df["duration_s"]
    return df


def _n_blocks(duration_s: float, block_sec: int) -> int:
    return max(1, int((max(float(duration_s), 0.0) + block_sec - 1e-9) // block_sec))


def find_turns_between(
    segments: pd.DataFrame,
    labels_a: Set[str],
    labels_b: Set[str],
    turn_type: str,
    gap_sec: float = TURN_GAP_SEC,
    block_sec: int = BLOCK_SEC,
) -> pd.DataFrame:
    """Return one row per turn between label sets A and B (either direction)."""
    keep = labels_a | labels_b
    sub = segments[segments["label"].isin(keep)].sort_values("start_time_s").reset_index(drop=True)
    rows: List[dict] = []
    for i in range(len(sub) - 1):
        a = sub.iloc[i]
        b = sub.iloc[i + 1]
        la, lb = str(a["label"]), str(b["label"])
        if la == lb:
            continue
        if not ((la in labels_a and lb in labels_b) or (la in labels_b and lb in labels_a)):
            continue
        gap = float(b["start_time_s"]) - float(a["end_time_s"])
        if gap < 0:
            gap = 0.0
        if gap > gap_sec:
            continue
        turn_time = float(b["start_time_s"])
        block = int(turn_time // block_sec)
        rows.append(
            {
                "uid": a["uid"],
                "turn_type": turn_type,
                "first_start": float(a["start_time_s"]),
                "first_end": float(a["end_time_s"]),
                "first_label": la,
                "second_start": float(b["start_time_s"]),
                "second_end": float(b["end_time_s"]),
                "second_label": lb,
                "gap_sec": gap,
                "direction": f"{la}->{lb}",
                "turn_time_s": turn_time,
                "block_5min": block,
                "block_start_s": block * block_sec,
                "block_end_s": block * block_sec + block_sec,
            }
        )
    return pd.DataFrame(rows)


def find_all_turns(
    segments: pd.DataFrame,
    gap_sec: float = TURN_GAP_SEC,
    block_sec: int = BLOCK_SEC,
) -> pd.DataFrame:
    parts = [
        find_turns_between(segments, a, b, turn_type, gap_sec, block_sec)
        for turn_type, a, b in TURN_SPECS
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def speech_duration_by_block(
    segments: pd.DataFrame,
    block_sec: int,
    duration_s: float,
) -> Dict[str, List[float]]:
    """Seconds of speech per label in each block (overlap clipped to block)."""
    n = _n_blocks(duration_s, block_sec)
    totals = {label: [0.0] * n for label in SPEECH_LABELS}
    for _, row in segments.iterrows():
        label = str(row["label"])
        if label not in totals:
            continue
        start = float(row["start_time_s"])
        end = float(row["end_time_s"])
        b0 = int(start // block_sec)
        b1 = int((end - 1e-9) // block_sec)
        for b in range(b0, min(b1, n - 1) + 1):
            b_start = b * block_sec
            b_end = b_start + block_sec
            overlap = max(0.0, min(end, b_end) - max(start, b_start))
            totals[label][b] += overlap
    return totals


def _turn_counts_for_block(turns: pd.DataFrame, block: int) -> dict:
    g = turns[turns["block_5min"] == block] if not turns.empty else turns
    adult = g[g["turn_type"] == "adult_kchi"] if not g.empty else g
    return {
        "turns_adult_kchi": int(len(adult)),
        "turns_FEM_kchi": int(((adult["first_label"] == "FEM") | (adult["second_label"] == "FEM")).sum())
        if len(adult)
        else 0,
        "turns_MAL_kchi": int(((adult["first_label"] == "MAL") | (adult["second_label"] == "MAL")).sum())
        if len(adult)
        else 0,
        "turns_mal_fem": int((g["turn_type"] == "mal_fem").sum()) if not g.empty else 0,
        "turns_kchi_och": int((g["turn_type"] == "kchi_och").sum()) if not g.empty else 0,
    }


def _turn_counts_total(turns: pd.DataFrame) -> dict:
    if turns.empty:
        return {
            "turns_adult_kchi": 0,
            "turns_FEM_kchi": 0,
            "turns_MAL_kchi": 0,
            "turns_mal_fem": 0,
            "turns_kchi_och": 0,
        }
    adult = turns[turns["turn_type"] == "adult_kchi"]
    return {
        "turns_adult_kchi": int(len(adult)),
        "turns_FEM_kchi": int(((adult["first_label"] == "FEM") | (adult["second_label"] == "FEM")).sum()),
        "turns_MAL_kchi": int(((adult["first_label"] == "MAL") | (adult["second_label"] == "MAL")).sum()),
        "turns_mal_fem": int((turns["turn_type"] == "mal_fem").sum()),
        "turns_kchi_och": int((turns["turn_type"] == "kchi_och").sum()),
    }


def _speech_duration_total(segments: pd.DataFrame) -> dict:
    out = {}
    for label in SPEECH_LABELS:
        sub = segments[segments["label"] == label]
        out[f"speech_duration_{label}_s"] = round(float(sub["duration_s"].sum()), 3) if len(sub) else 0.0
    return out


def summarize(
    segments: pd.DataFrame,
    turns: pd.DataFrame,
    uid: str,
    block_sec: int = BLOCK_SEC,
    duration_s: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-block (zero-filled) and total summary tables."""
    n_blocks = _n_blocks(duration_s, block_sec)
    if not turns.empty:
        n_blocks = max(n_blocks, int(turns["block_5min"].max()) + 1)

    speech = speech_duration_by_block(segments, block_sec, duration_s)

    block_rows = []
    for b in range(n_blocks):
        row = {
            "uid": uid,
            "block_5min": b,
            "block_start_s": b * block_sec,
            "block_end_s": b * block_sec + block_sec,
            **_turn_counts_for_block(turns, b),
        }
        for label in SPEECH_LABELS:
            row[f"speech_duration_{label}_s"] = round(speech[label][b], 3)
        block_rows.append(row)

    blocks = pd.DataFrame(block_rows)
    total = pd.DataFrame(
        [
            {
                "uid": uid,
                **_turn_counts_total(turns),
                **_speech_duration_total(segments),
            }
        ]
    )
    return blocks, total


def compute_turns(
    rttm_path: Path | str,
    out_dir: Path | str,
    gap_sec: float = TURN_GAP_SEC,
    block_sec: int = BLOCK_SEC,
    uids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load rttm, compute turns + speech duration, write CSVs."""
    rttm_path = Path(rttm_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_rttm(rttm_path)
    if uids:
        df = df[df["uid"].isin(uids)].copy()
        if df.empty:
            raise ValueError(f"No rows for uid filter {uids}")

    all_turns: List[pd.DataFrame] = []
    block_parts: List[pd.DataFrame] = []
    total_parts: List[pd.DataFrame] = []
    for uid, sub in df.groupby("uid", sort=True):
        turns = find_all_turns(sub, gap_sec=float(gap_sec), block_sec=int(block_sec))
        if not turns.empty:
            all_turns.append(turns)
        duration_s = float(sub["end_time_s"].max()) if len(sub) else 0.0
        blocks, total = summarize(sub, turns, str(uid), block_sec=int(block_sec), duration_s=duration_s)
        block_parts.append(blocks)
        total_parts.append(total)

    turns_df = pd.concat(all_turns, ignore_index=True) if all_turns else pd.DataFrame()
    blocks_df = pd.concat(block_parts, ignore_index=True) if block_parts else pd.DataFrame()
    total_df = pd.concat(total_parts, ignore_index=True) if total_parts else pd.DataFrame()

    turns_df.to_csv(out / "conversational_turns_detail.csv", index=False)
    blocks_df.to_csv(out / "conversational_turns_by_5min.csv", index=False)
    total_df.to_csv(out / "conversational_turns_total.csv", index=False)
    return turns_df, blocks_df, total_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational turns + speech duration from VTC rttm.csv")
    parser.add_argument("--rttm", required=True, help="Path to rttm.csv from VTC")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gap-sec", type=float, default=TURN_GAP_SEC)
    parser.add_argument("--block-sec", type=int, default=BLOCK_SEC)
    args = parser.parse_args()

    turns_df, blocks_df, total_df = compute_turns(
        args.rttm,
        args.out_dir,
        gap_sec=float(args.gap_sec),
        block_sec=int(args.block_sec),
    )
    print(f"Gap <= {args.gap_sec}s; blocks = {args.block_sec}s")
    print(f"Detail turns: {len(turns_df)}")
    print(f"Wrote CSVs under {args.out_dir}")
    if not total_df.empty:
        print(total_df.to_string(index=False))
    if not blocks_df.empty:
        print(blocks_df.to_string(index=False))


if __name__ == "__main__":
    main()
