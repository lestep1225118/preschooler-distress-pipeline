"""
Phase 2 — after human review of peak windows.

Reads a filled review_notes_*.csv (is_true_distress = 1/0), then:
  1) Keeps reviewed-true peak windows
  2) Expands each true segment to a per-second distress timeline
     (segment_start .. segment_end-1), matching the lab step
     "recreate a file with human-reviewed distress seconds with times"
  3) Merges those into episodes (default 5 min gap)
  4) Optionally cuts episode clips with a pre-buffer (default 60 s)

Usage:
  python -m lab_pipeline.run_phase2_episodes --config lab_pipeline/config.yaml \\
      --review-notes lab_runs/example_run/output/review_notes_consecutive.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from lab_pipeline import (
    ensure_dir,
    load_config,
    resolve_audio_map,
)


def _parse_true(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "y", "distress", "cry"}


def expand_true_segments_to_seconds(true_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per human-accepted distress second (within reviewed-true segments)."""
    rows: List[dict] = []
    for _, r in true_rows.iterrows():
        # Prefer full segment span when present; else peak window only
        if "segment_start" in true_rows.columns and pd.notna(r.get("segment_start")):
            start = int(r["segment_start"])
            end = int(r["segment_end"]) if pd.notna(r.get("segment_end")) else start + 5
        else:
            start = int(r["peak_window_start"])
            end = int(r["peak_window_end"]) if pd.notna(r.get("peak_window_end")) else start + 5
        pid = r.get("participant_id", "")
        fid = str(r["file_id"])
        seg_id = r.get("segment_id", "")
        for sec in range(start, end):
            rows.append(
                {
                    "participant_id": pid,
                    "file_id": fid,
                    "segment_id": seg_id,
                    "second": sec,
                    "source": "reviewed_true_segment",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["file_id", "second"]).sort_values(
        ["file_id", "second"]
    )


def episode_bounds_from_seconds(
    seconds: List[int], gap_sec: int
) -> List[Tuple[int, int]]:
    """Merge distress seconds into episodes if gap between seconds <= gap_sec."""
    if not seconds:
        return []
    ordered = sorted(seconds)
    episodes: List[List[int]] = [[ordered[0]]]
    for s in ordered[1:]:
        if s - episodes[-1][-1] <= gap_sec:
            episodes[-1].append(s)
        else:
            episodes.append([s])
    return [(ep[0], ep[-1] + 1) for ep in episodes]  # end exclusive


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab pipeline Phase 2: episodes from review notes")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--review-notes",
        required=True,
        help="Filled review_notes CSV with is_true_distress column",
    )
    parser.add_argument("--write-clips", action="store_true", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_dir(Path(cfg["run_dir"]))
    out_dir = ensure_dir(run_dir / "output" / "phase2")

    notes = pd.read_csv(args.review_notes)
    if "is_true_distress" not in notes.columns:
        raise SystemExit("review notes must include is_true_distress column")

    notes["is_true"] = notes["is_true_distress"].map(_parse_true)
    true_rows = notes[notes["is_true"]].copy()
    true_path = out_dir / "human_reviewed_distress_windows.csv"
    true_rows.to_csv(true_path, index=False)
    print(f"True-distress reviewed segments: {len(true_rows)} -> {true_path}")

    seconds_df = expand_true_segments_to_seconds(true_rows)
    sec_path = out_dir / "human_reviewed_distress_seconds.csv"
    seconds_df.to_csv(sec_path, index=False)
    print(f"Human-reviewed distress seconds: {len(seconds_df)} -> {sec_path}")

    gap_sec = int(cfg.get("episode_gap_sec", 300))
    pre_buf = int(cfg.get("episode_pre_buffer_sec", 60))
    write_clips = args.write_clips
    if write_clips is None:
        write_clips = bool(cfg.get("write_clips", True))

    episode_rows: List[dict] = []
    if seconds_df.empty:
        episodes = pd.DataFrame(episode_rows)
    else:
        for file_id, sub in seconds_df.groupby("file_id", sort=True):
            secs = [int(x) for x in sub["second"].tolist()]
            eps = episode_bounds_from_seconds(secs, gap_sec=gap_sec)
            pid = sub["participant_id"].iloc[0] if "participant_id" in sub.columns else file_id
            for i, (estart, eend) in enumerate(eps, start=1):
                clip_start = max(0, estart - pre_buf)
                n_sec = sum(1 for s in secs if estart <= s < eend)
                episode_rows.append(
                    {
                        "participant_id": pid,
                        "file_id": str(file_id),
                        "episode_id": i,
                        "episode_start": estart,
                        "episode_end": eend,
                        "episode_span_sec": eend - estart,
                        "clip_start": clip_start,
                        "clip_end": eend,
                        "pre_buffer_sec": estart - clip_start,
                        "n_distress_seconds": n_sec,
                    }
                )
        episodes = pd.DataFrame(episode_rows)

    ep_path = out_dir / "episodes.csv"
    episodes.to_csv(ep_path, index=False)
    print(f"Episodes (gap={gap_sec}s): {len(episodes)} -> {ep_path}")

    if episodes.empty or not write_clips:
        print("Skipping episode clips.")
        return

    file_ids = sorted(episodes["file_id"].astype(str).unique())
    audio_map = resolve_audio_map(cfg, file_ids)
    if not audio_map:
        print("WARNING: no audio resolved; skipping clips.")
        return

    import soundfile as sf

    from lab_pipeline.audio_io import read_wav_segment

    clips_dir = ensure_dir(out_dir / "episode_clips")
    clip_paths = []
    statuses = []
    for _, row in episodes.iterrows():
        wav = audio_map.get(str(row["file_id"]))
        if not wav or not os.path.isfile(wav):
            clip_paths.append(None)
            statuses.append("missing_wav")
            continue
        try:
            audio, sr = read_wav_segment(wav, int(row["clip_start"]), int(row["clip_end"]))
            name = (
                f"{row['file_id']}_ep{int(row['episode_id']):03d}_"
                f"{int(row['clip_start'])}-{int(row['clip_end'])}.wav"
            )
            path = clips_dir / name
            sf.write(str(path), audio, sr)
            clip_paths.append(str(path))
            statuses.append("ok")
        except Exception as e:  # noqa: BLE001
            clip_paths.append(None)
            statuses.append(f"error:{e}")

    episodes["clip_path"] = clip_paths
    episodes["clip_status"] = statuses
    episodes.to_csv(ep_path, index=False)
    print(f"Episode clips -> {clips_dir}")


if __name__ == "__main__":
    main()
