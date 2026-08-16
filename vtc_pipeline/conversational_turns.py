"""
Count adult <-> key-child conversational turns from VTC (rttm.csv) output.

Definition (Dr. Ravindran):
  A conversational turn = speech by one party followed by speech by the other
  within TURN_GAP_SEC seconds (gap from end of first to start of second).

Parties:
  - Adult: FEM or MAL
  - Key child: KCHI
  - OCH is ignored for these turn counts

Reports:
  - Per 5-minute block (zero-filled for empty blocks)
  - Total turns for the full recording
  - Breakdown: adult-KCHI, FEM-KCHI, MAL-KCHI

Usage:
  python -m vtc_pipeline.conversational_turns --rttm path/to/rttm.csv --out-dir out/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

TURN_GAP_SEC = 5.0
BLOCK_SEC = 300  # 5 minutes
ADULT = {"FEM", "MAL"}
KEEP = {"FEM", "MAL", "KCHI"}


def load_rttm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"uid", "start_time_s", "duration_s", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"rttm CSV missing columns: {missing}")
    df = df.copy()
    df["end_time_s"] = df["start_time_s"] + df["duration_s"]
    return df


def role(label: str) -> Optional[str]:
    if label in ADULT:
        return "ADULT"
    if label == "KCHI":
        return "KCHI"
    return None


def find_turns(
    segments: pd.DataFrame,
    gap_sec: float = TURN_GAP_SEC,
    block_sec: int = BLOCK_SEC,
) -> pd.DataFrame:
    """Return one row per adult <-> KCHI turn."""
    sub = segments[segments["label"].isin(KEEP)].sort_values("start_time_s").reset_index(drop=True)
    rows: List[dict] = []
    for i in range(len(sub) - 1):
        a = sub.iloc[i]
        b = sub.iloc[i + 1]
        ra, rb = role(str(a["label"])), role(str(b["label"]))
        if ra is None or rb is None or ra == rb:
            continue
        gap = float(b["start_time_s"]) - float(a["end_time_s"])
        if gap < 0:
            gap = 0.0
        if gap > gap_sec:
            continue
        adult_label = str(a["label"]) if ra == "ADULT" else str(b["label"])
        turn_time = float(b["start_time_s"])
        rows.append(
            {
                "uid": a["uid"],
                "first_start": float(a["start_time_s"]),
                "first_end": float(a["end_time_s"]),
                "first_label": str(a["label"]),
                "second_start": float(b["start_time_s"]),
                "second_end": float(b["end_time_s"]),
                "second_label": str(b["label"]),
                "gap_sec": gap,
                "direction": f"{a['label']}->{b['label']}",
                "adult_label": adult_label,
                "turn_time_s": turn_time,
                "block_5min": int(turn_time // block_sec),
                "block_start_s": int(turn_time // block_sec) * block_sec,
                "block_end_s": int(turn_time // block_sec) * block_sec + block_sec,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    turns: pd.DataFrame,
    uid: str,
    block_sec: int = BLOCK_SEC,
    duration_s: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-block (zero-filled) and total summary tables."""

    def counts(g: Optional[pd.DataFrame]) -> dict:
        if g is None or len(g) == 0:
            return {
                "turns_adult_kchi": 0,
                "turns_FEM_kchi": 0,
                "turns_MAL_kchi": 0,
            }
        return {
            "turns_adult_kchi": int(len(g)),
            "turns_FEM_kchi": int((g["adult_label"] == "FEM").sum()),
            "turns_MAL_kchi": int((g["adult_label"] == "MAL").sum()),
        }

    by_block = {}
    if not turns.empty:
        for b, g in turns.groupby("block_5min", sort=True):
            by_block[int(b)] = counts(g)

    n_blocks = max(1, int((max(float(duration_s), 0.0) + block_sec - 1e-9) // block_sec))
    if by_block:
        n_blocks = max(n_blocks, max(by_block.keys()) + 1)

    block_rows = []
    for b in range(n_blocks):
        c = by_block.get(b, counts(None))
        block_rows.append(
            {
                "uid": uid,
                "block_5min": b,
                "block_start_s": b * block_sec,
                "block_end_s": b * block_sec + block_sec,
                **c,
            }
        )
    blocks = pd.DataFrame(block_rows)
    total = pd.DataFrame([{"uid": uid, **counts(turns if not turns.empty else None)}])
    return blocks, total


def compute_turns(
    rttm_path: Path | str,
    out_dir: Path | str,
    gap_sec: float = TURN_GAP_SEC,
    block_sec: int = BLOCK_SEC,
    uids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load rttm, compute turns, write CSVs. Returns (detail, by_5min, total)."""
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
        turns = find_turns(sub, gap_sec=float(gap_sec), block_sec=int(block_sec))
        if not turns.empty:
            all_turns.append(turns)
        duration_s = float(sub["end_time_s"].max()) if len(sub) else 0.0
        blocks, total = summarize(
            turns, str(uid), block_sec=int(block_sec), duration_s=duration_s
        )
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
    parser = argparse.ArgumentParser(description="Adult-KCHI conversational turns from VTC rttm.csv")
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
