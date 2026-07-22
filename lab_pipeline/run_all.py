"""
One-command lab pipeline runner.

Steps:
  1) (optional) AF3 score folder of WAVs  --requires GPU + adapter
  2) Phase 1: review CSVs + 5 s peak clips
  3) (optional) Phase 2: episodes from filled review notes

Examples:
  # Local post-process only (scores already exist):
  python -m lab_pipeline.run_all --config lab_pipeline/config.yaml

  # Full path on Sapelo2 after scoring:
  python -m lab_pipeline.run_all --config lab_pipeline/config.yaml --score

  # After RAs fill notes:
  python -m lab_pipeline.run_all --config lab_pipeline/config.yaml \\
    --phase2 --review-notes lab_runs/my_run/output/review_notes_consecutive.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lab_pipeline import load_config


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lab distress pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run AF3 folder scoring first (needs GPU + adapter_dir in config)",
    )
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 episode merge")
    parser.add_argument(
        "--review-notes",
        default=None,
        help="Filled review notes CSV (required with --phase2)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    py = sys.executable

    if args.score:
        score_cfg = cfg.get("scoring") or {}
        audio_dir = score_cfg.get("audio_dir") or cfg.get("audio_dir")
        audio_map = score_cfg.get("audio_map_csv") or cfg.get("audio_map_csv")
        adapter = score_cfg.get("adapter_dir")
        out_csv = score_cfg.get("out_csv") or str(
            Path(cfg["run_dir"]) / "output" / "windows_scored.csv"
        )
        if not audio_dir and not audio_map:
            raise SystemExit("scoring needs audio_dir or audio_map_csv for --score")
        if not adapter and not score_cfg.get("base_only"):
            raise SystemExit("scoring.adapter_dir required for --score (or scoring.base_only: true)")

        cmd = [
            py,
            "-m",
            "lab_pipeline.score_audio_folder",
            "--out-csv",
            str(out_csv),
            "--window-sec",
            str(cfg.get("window_sec", 5)),
            "--stride",
            str(score_cfg.get("stride", cfg.get("window_sec", 5))),
            "--resume",
        ]
        if audio_map:
            cmd.extend(["--audio-map-csv", str(audio_map)])
        elif audio_dir:
            cmd.extend(["--audio-dir", str(audio_dir)])
        if score_cfg.get("base_only"):
            cmd.append("--base-only")
        else:
            cmd.extend(["--adapter-dir", str(adapter)])
        if score_cfg.get("recursive"):
            cmd.append("--recursive")
        if score_cfg.get("limit_windows_per_file"):
            cmd.extend(
                ["--limit-windows-per-file", str(score_cfg["limit_windows_per_file"])]
            )
        _run(cmd)

    if not args.skip_phase1:
        _run([py, "-m", "lab_pipeline.run_phase1_review", "--config", args.config])

    if args.phase2:
        if not args.review_notes:
            raise SystemExit("--phase2 requires --review-notes")
        _run(
            [
                py,
                "-m",
                "lab_pipeline.run_phase2_episodes",
                "--config",
                args.config,
                "--review-notes",
                args.review_notes,
            ]
        )

    print("Pipeline finished.", flush=True)


if __name__ == "__main__":
    main()
