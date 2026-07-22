"""
Phase 1 — local post-processing after AF3 scoring.

Inputs (via config):
  - AF3 window scores CSV (start_second, pred_score, file_id or participant_id)
  - optional audio_map / audio_dir for clipping

Outputs under {run_dir}/output/:
  - windows_scored.csv                          (all 5 s epochs + pred_score)
  - positive_windows.csv                        (epochs >= threshold, with timestamps)
  - review_consecutive.csv                      (merged adjacent positives + peak window)
  - review_cluster_top50.csv                    (30 s clusters, top 50%, peak window)
  - review_notes_consecutive.csv                (RA template)
  - review_notes_cluster_top50.csv
  - clips_consecutive/                          (5 s peak WAVs, if write_clips)
  - clips_cluster_top50/

Usage:
  python -m lab_pipeline.run_phase1_review --config lab_pipeline/config.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from lab_pipeline import (
    build_segment_table,
    ensure_dir,
    load_config,
    load_scores,
    make_review_notes_template,
    resolve_audio_map,
    write_peak_clips,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab pipeline Phase 1: review CSVs + clips")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_dir(Path(cfg["run_dir"]))
    out_dir = ensure_dir(run_dir / "output")
    # Keep a copy of the config used for this run
    shutil.copy2(args.config, run_dir / "config_used.yaml")

    scores = load_scores(cfg["scores_csv"])
    scores_out = out_dir / "windows_scored.csv"
    scores.to_csv(scores_out, index=False)
    print(f"Loaded {len(scores)} windows across {scores['file_id'].nunique()} files")
    print(f"Wrote {scores_out}")

    threshold = float(cfg.get("threshold", 0.32))
    window_sec = int(cfg.get("window_sec", 5))
    cluster_gap = int(cfg.get("cluster_gap_sec", 30))
    top_fraction = float(cfg.get("top_fraction", 0.50))
    modes = cfg.get("modes") or {"consecutive": True, "cluster_top50": True}
    write_clips = bool(cfg.get("write_clips", True))

    pos = scores[scores["pred_score"].astype(float) >= threshold].copy()
    pos = pos.sort_values(["file_id", "start_second"])
    pos_path = out_dir / "positive_windows.csv"
    pos.to_csv(pos_path, index=False)
    print(f"Positive windows (>= {threshold}): {len(pos)} -> {pos_path}")

    file_ids = sorted(scores["file_id"].astype(str).unique())
    audio_map = resolve_audio_map(cfg, file_ids)
    if write_clips and not audio_map:
        print(
            "WARNING: write_clips=true but no wavs resolved. "
            "Set audio_dir or audio_map_csv. Continuing without clips."
        )
        write_clips = False
    elif write_clips:
        print(f"Resolved audio for {len(audio_map)}/{len(file_ids)} files")

    def emit(name: str, segments: pd.DataFrame) -> None:
        if segments.empty:
            print(f"[{name}] no positive segments at threshold={threshold}")
            segments.to_csv(out_dir / f"review_{name}.csv", index=False)
            return
        if write_clips:
            clips_dir = ensure_dir(out_dir / f"clips_{name}")
            segments = write_peak_clips(segments, audio_map, clips_dir)
        else:
            segments = segments.copy()
            segments["wav_path"] = segments["file_id"].map(audio_map)
            segments["clip_path"] = None
            segments["clip_status"] = "skipped"

        review_path = out_dir / f"review_{name}.csv"
        segments.to_csv(review_path, index=False)
        notes = make_review_notes_template(segments)
        notes_path = out_dir / f"review_notes_{name}.csv"
        notes.to_csv(notes_path, index=False)
        print(f"[{name}] {len(segments)} segments -> {review_path}")
        print(f"[{name}] review notes template -> {notes_path}")

    if modes.get("consecutive", True):
        consec = build_segment_table(
            scores,
            threshold,
            mode="consecutive",
            window_sec=window_sec,
            top_fraction=None,
        )
        emit("consecutive", consec)

    if modes.get("cluster_top50", True):
        clusters = build_segment_table(
            scores,
            threshold,
            mode="cluster",
            window_sec=window_sec,
            cluster_gap_sec=cluster_gap,
            top_fraction=top_fraction,
        )
        emit("cluster_top50", clusters)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
