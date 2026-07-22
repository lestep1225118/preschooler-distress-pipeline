"""
Lab GUI for the AF3 distress-detection pipeline (cross-platform).

Wraps the existing folder/config pipeline so RAs/staff can:
  1) Point at audio + scores (or see Sapelo2 scoring instructions)
  2) Run Phase 1 (both review strategies + 5 s clips)
  3) Fill / save review notes
  4) Run Phase 2 (distress seconds → episodes → buffered clips)

Launch:
  python -m lab_pipeline.gui_app
  # or: lab_pipeline/launch_gui.ps1
"""

from __future__ import annotations

import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _write_config(
    run_dir: str,
    scores_csv: str,
    audio_dir: str,
    audio_map_csv: str,
    threshold: float,
    consecutive: bool,
    cluster_top50: bool,
    write_clips: bool,
    top_fraction: float,
    episode_gap_sec: int,
    episode_pre_buffer_sec: int,
) -> Path:
    run_path = _rel(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    cfg: Dict[str, Any] = {
        "run_dir": str(run_path),
        "scores_csv": str(_rel(scores_csv)) if scores_csv else "",
        "audio_map_csv": str(_rel(audio_map_csv)) if audio_map_csv else None,
        "audio_dir": str(_rel(audio_dir)) if audio_dir else None,
        "audio_name_template": "{file_id}_spliced_6hours.wav",
        "threshold": float(threshold),
        "modes": {
            "consecutive": bool(consecutive),
            "cluster_top50": bool(cluster_top50),
        },
        "cluster_gap_sec": 30,
        "top_fraction": float(top_fraction),
        "window_sec": 5,
        "write_clips": bool(write_clips),
        "episode_gap_sec": int(episode_gap_sec),
        "episode_pre_buffer_sec": int(episode_pre_buffer_sec),
        "scoring": {
            "audio_dir": str(_rel(audio_dir)) if audio_dir else None,
            "adapter_dir": None,
            "out_csv": str(run_path / "output" / "windows_scored.csv"),
            "stride": 5,
            "recursive": True,
            "base_only": False,
        },
    }
    # yaml nulls: drop empty optional paths
    if not cfg["audio_map_csv"]:
        cfg["audio_map_csv"] = None
    if not cfg["audio_dir"]:
        cfg["audio_dir"] = None

    cfg_path = run_path / "config_gui.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def build_audio_map(audio_dir: str, run_dir: str, recursive: bool = False) -> Tuple[str, str]:
    try:
        from lab_pipeline.build_audio_map import participant_from_stem

        root = _rel(audio_dir)
        if not root.is_dir():
            return "", f"Audio folder not found: {root}"

        # Prefer daylong spliced files at this folder level when present
        spliced = sorted(root.glob("*_spliced_6hours.wav"))
        if spliced:
            wavs = spliced
            note = "using *_spliced_6hours.wav in folder"
        else:
            wavs = sorted(root.rglob("*.wav") if recursive else root.glob("*.wav"))
            note = "recursive" if recursive else "top-level only"

        if not wavs:
            return "", f"No WAV files under {root} ({note})"
        if len(wavs) > 5000:
            return (
                "",
                f"Found {len(wavs)} WAVs ({note}) — refusing to map that many. "
                "Point audio folder at the daylong recordings only, or use non-recursive.",
            )

        rows = [
            {
                "participant_id": participant_from_stem(w.stem),
                "file_id": w.stem.replace("_spliced_6hours", "")
                if w.stem.endswith("_spliced_6hours")
                else w.stem,
                "wav_path": str(w.resolve()),
            }
            for w in wavs
        ]
        # For spliced files, file_id should match scores (participant id string)
        out = _rel(run_dir) / "audio_map.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        return (
            str(out),
            f"Mapped {len(rows)} WAV(s) across {pd.DataFrame(rows)['participant_id'].nunique()} "
            f"participant(s) ({note}).",
        )
    except Exception:
        return "", traceback.format_exc()


def run_phase1(
    run_dir: str,
    scores_csv: str,
    audio_dir: str,
    audio_map_csv: str,
    threshold: float,
    consecutive: bool,
    cluster_top50: bool,
    write_clips: bool,
    top_fraction: float,
    episode_gap_sec: int,
    episode_pre_buffer_sec: int,
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        if not scores_csv or not _rel(scores_csv).is_file():
            return (
                "Scores CSV not found. Point to an AF3 windows CSV "
                "(from Sapelo2 scoring) or the existing paper scores.",
                None,
                None,
                None,
            )
        if not consecutive and not cluster_top50:
            return "Enable at least one review strategy.", None, None, None

        cfg_path = _write_config(
            run_dir,
            scores_csv,
            audio_dir,
            audio_map_csv,
            threshold,
            consecutive,
            cluster_top50,
            write_clips,
            top_fraction,
            episode_gap_sec,
            episode_pre_buffer_sec,
        )
        # Import after config write; call main via subprocess-like API
        from lab_pipeline.run_phase1_review import main as phase1_main
        import sys

        old = sys.argv
        try:
            sys.argv = ["run_phase1_review", "--config", str(cfg_path)]
            phase1_main()
        finally:
            sys.argv = old

        out = _rel(run_dir) / "output"
        pos = pd.read_csv(out / "positive_windows.csv") if (out / "positive_windows.csv").is_file() else None
        consec = (
            pd.read_csv(out / "review_consecutive.csv")
            if (out / "review_consecutive.csv").is_file()
            else None
        )
        clust = (
            pd.read_csv(out / "review_cluster_top50.csv")
            if (out / "review_cluster_top50.csv").is_file()
            else None
        )
        n_c = 0
        n_k = 0
        if (out / "clips_consecutive").is_dir():
            n_c = len(list((out / "clips_consecutive").glob("*.wav")))
        if (out / "clips_cluster_top50").is_dir():
            n_k = len(list((out / "clips_cluster_top50").glob("*.wav")))
        msg = (
            f"Phase 1 complete.\n"
            f"Output: {out}\n"
            f"Positive windows: {0 if pos is None else len(pos)}\n"
            f"Consecutive segments: {0 if consec is None else len(consec)} (clips: {n_c})\n"
            f"Cluster top-50% segments: {0 if clust is None else len(clust)} (clips: {n_k})\n"
            f"Fill review notes next (Review notes tab)."
        )
        # Preview only — full CSVs stay on disk
        pos_prev = None if pos is None else pos.head(100)
        consec_prev = None if consec is None else consec.head(100)
        clust_prev = None if clust is None else clust.head(100)
        return msg, pos_prev, consec_prev, clust_prev
    except Exception:
        return traceback.format_exc(), None, None, None


def load_notes(run_dir: str, mode: str) -> Tuple[Optional[pd.DataFrame], str]:
    path = _rel(run_dir) / "output" / f"review_notes_{mode}.csv"
    if not path.is_file():
        return None, f"Notes not found: {path}\nRun Phase 1 first."
    df = pd.read_csv(path)
    # Ensure editable columns exist
    for col in ("is_true_distress", "reviewer", "notes"):
        if col not in df.columns:
            df[col] = ""
    return df, f"Loaded {len(df)} rows from {path}"


def save_notes(run_dir: str, mode: str, df: pd.DataFrame, reviewer: str) -> str:
    try:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return "No notes table to save."
        out = df.copy()
        if reviewer:
            # only fill blank reviewer cells
            if "reviewer" not in out.columns:
                out["reviewer"] = reviewer
            else:
                blank = out["reviewer"].isna() | (out["reviewer"].astype(str).str.strip() == "")
                out.loc[blank, "reviewer"] = reviewer
        path = _rel(run_dir) / "output" / f"review_notes_{mode}.csv"
        # also keep a timestamped filled copy
        filled = _rel(run_dir) / "output" / f"review_notes_{mode}_filled.csv"
        out.to_csv(path, index=False)
        out.to_csv(filled, index=False)
        n_true = 0
        if "is_true_distress" in out.columns:
            n_true = int(
                out["is_true_distress"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["1", "true", "yes", "y"])
                .sum()
            )
        return f"Saved {len(out)} rows ({n_true} marked true) -> {path}"
    except Exception:
        return traceback.format_exc()


def run_phase2(
    run_dir: str,
    mode: str,
    scores_csv: str,
    audio_dir: str,
    audio_map_csv: str,
    threshold: float,
    consecutive: bool,
    cluster_top50: bool,
    write_clips: bool,
    top_fraction: float,
    episode_gap_sec: int,
    episode_pre_buffer_sec: int,
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        notes = _rel(run_dir) / "output" / f"review_notes_{mode}.csv"
        if not notes.is_file():
            return f"Notes not found: {notes}", None, None

        cfg_path = _write_config(
            run_dir,
            scores_csv,
            audio_dir,
            audio_map_csv,
            threshold,
            consecutive,
            cluster_top50,
            write_clips,
            top_fraction,
            episode_gap_sec,
            episode_pre_buffer_sec,
        )
        from lab_pipeline.run_phase2_episodes import main as phase2_main
        import sys

        old = sys.argv
        try:
            sys.argv = [
                "run_phase2_episodes",
                "--config",
                str(cfg_path),
                "--review-notes",
                str(notes),
            ]
            phase2_main()
        finally:
            sys.argv = old

        p2 = _rel(run_dir) / "output" / "phase2"
        secs = (
            pd.read_csv(p2 / "human_reviewed_distress_seconds.csv")
            if (p2 / "human_reviewed_distress_seconds.csv").is_file()
            else None
        )
        eps = pd.read_csv(p2 / "episodes.csv") if (p2 / "episodes.csv").is_file() else None
        n_clips = 0
        if (p2 / "episode_clips").is_dir():
            n_clips = len(list((p2 / "episode_clips").glob("*.wav")))
        msg = (
            f"Phase 2 complete.\n"
            f"Output: {p2}\n"
            f"Distress seconds: {0 if secs is None else len(secs)}\n"
            f"Episodes: {0 if eps is None else len(eps)}\n"
            f"Episode clips: {n_clips}"
        )
        secs_prev = None if secs is None else secs.head(100)
        eps_prev = None if eps is None else eps.head(100)
        return msg, secs_prev, eps_prev
    except Exception:
        return traceback.format_exc(), None, None


def zip_folder(folder: Path, zip_path: Path) -> Optional[str]:
    if not folder.is_dir():
        return None
    base = str(zip_path.with_suffix(""))
    shutil.make_archive(base, "zip", root_dir=folder)
    return str(zip_path)


def make_download_zips(run_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    out = _rel(run_dir) / "output"
    if not out.is_dir():
        return None, None, None, "No output folder yet."
    dl = out / "_downloads"
    dl.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    c1 = out / "clips_consecutive"
    c2 = out / "clips_cluster_top50"
    e = out / "phase2" / "episode_clips"
    z1 = zip_folder(c1, dl / f"clips_consecutive_{stamp}.zip") if c1.is_dir() else None
    z2 = zip_folder(c2, dl / f"clips_cluster_top50_{stamp}.zip") if c2.is_dir() else None
    z3 = zip_folder(e, dl / f"episode_clips_{stamp}.zip") if e.is_dir() else None
    return z1, z2, z3, f"Downloads folder: {dl}"


def gpu_status() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"GPU available: {name} ({mem:.1f} GB). Local AF3 scoring can run if adapter path is set."
        return "No CUDA GPU detected. Use Sapelo2 for AF3 scoring, or install a CUDA torch build."
    except Exception as e:  # noqa: BLE001
        return f"Could not check GPU ({e}). Sapelo2 scoring still works."


def run_af3_score(
    run_dir: str,
    audio_dir: str,
    audio_map_csv: str,
    adapter_dir: str,
    base_only: bool,
    limit_files: float,
    limit_windows: float,
    resume: bool,
) -> Tuple[str, str]:
    """
    Score audio with AF3. Returns (status_message, scores_csv_path).
    scores_csv_path is empty on failure.
    """
    try:
        out_csv = _rel(run_dir) / "output" / "windows_scored.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Ensure we have an audio map (preferred — avoids scanning huge trees)
        map_path = audio_map_csv.strip() if audio_map_csv else ""
        if not map_path or not _rel(map_path).is_file():
            if not audio_dir:
                return "Set Audio folder or Audio map CSV first.", ""
            map_path, map_msg = build_audio_map(audio_dir, run_dir, recursive=False)
            if not map_path:
                return f"Could not build audio map.\n{map_msg}", ""
        else:
            map_path = str(_rel(map_path))
            map_msg = f"Using existing map: {map_path}"

        if not base_only:
            if not adapter_dir or not _rel(adapter_dir).is_dir():
                return (
                    "LoRA adapter folder not found.\n"
                    "Set Adapter dir to a folder containing adapter_config.json "
                    "(copy from Sapelo2/Drive), or check 'Base AF3 only' for a "
                    "non-finetuned smoke test.\n"
                    f"{map_msg}",
                    "",
                )
            if not (_rel(adapter_dir) / "adapter_config.json").is_file():
                return (
                    f"No adapter_config.json in {adapter_dir}. "
                    "Point at the lora_adapter folder itself.",
                    "",
                )

        import sys

        from lab_pipeline.score_audio_folder import main as score_main

        cmd = [
            "score_audio_folder",
            "--audio-map-csv",
            map_path,
            "--out-csv",
            str(out_csv),
        ]
        if resume:
            cmd.append("--resume")
        if base_only:
            cmd.append("--base-only")
        else:
            cmd.extend(["--adapter-dir", str(_rel(adapter_dir))])
        lf = int(limit_files) if limit_files and float(limit_files) > 0 else None
        lw = int(limit_windows) if limit_windows and float(limit_windows) > 0 else None
        if lf:
            cmd.extend(["--limit-files", str(lf)])
        if lw:
            cmd.extend(["--limit-windows-per-file", str(lw)])

        old = sys.argv
        try:
            sys.argv = cmd
            score_main()
        finally:
            sys.argv = old

        if not out_csv.is_file():
            return "Scoring finished but no CSV was written.", ""
        n = len(pd.read_csv(out_csv))
        msg = (
            f"AF3 scoring complete.\n{map_msg}\n"
            f"Wrote {n} windows -> {out_csv}\n"
            "Scores CSV field updated. Continue to Phase 1."
        )
        return msg, str(out_csv)
    except Exception:
        return traceback.format_exc(), ""


def prepare_sapelo2_bundle(run_dir: str, audio_dir: str, audio_map_csv: str) -> str:
    """Write a Sapelo2 submit helper + wav list into the run folder."""
    try:
        run_path = _rel(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        bundle = run_path / "sapelo2_bundle"
        bundle.mkdir(parents=True, exist_ok=True)

        map_path = audio_map_csv.strip() if audio_map_csv else ""
        if not map_path or not _rel(map_path).is_file():
            map_path, map_msg = build_audio_map(audio_dir, run_dir, recursive=False)
            if not map_path:
                return f"Could not build audio map.\n{map_msg}"
        else:
            map_path = str(_rel(map_path))
            map_msg = f"Using map {map_path}"

        am = pd.read_csv(map_path)
        wav_list = bundle / "wav_list.txt"
        with open(wav_list, "w", encoding="utf-8") as f:
            for p in am["wav_path"].astype(str):
                f.write(p.replace("\\", "/") + "\n")

        shutil.copy2(map_path, bundle / "audio_map.csv")

        submit = bundle / "submit_on_sapelo2.sh"
        submit.write_text(
            """#!/bin/bash
# Run on Sapelo2 after syncing this repo and uploading WAVs / audio_map.
set -euo pipefail
export PROJ=/scratch/ls80136/infant_distress
export CODE=$PROJ/code/InfantDistressClassification
export PY=$PROJ/env/af3/bin/python
export ADAPTER=${ADAPTER:-$PROJ/models/fold_105/lora_adapter}
export OUT_CSV=${OUT_CSV:-$PROJ/lab_incoming/scores/windows_scored.csv}
export MAP=${MAP:-$PROJ/lab_incoming/audio_map.csv}

cd "$CODE"
$PY -m lab_pipeline.score_audio_folder \\
  --audio-map-csv "$MAP" \\
  --adapter-dir "$ADAPTER" \\
  --out-csv "$OUT_CSV" \\
  --resume

echo "Download $OUT_CSV and set it as Scores CSV in the GUI, then Run Phase 1."
""",
            encoding="utf-8",
        )

        readme = bundle / "README.txt"
        readme.write_text(
            f"""Sapelo2 scoring bundle
======================
{map_msg}
Files: {len(am)}

1) Upload WAVs to Sapelo2 (keep separate files; do not merge).
2) Upload audio_map.csv (paths must be updated to Sapelo2 paths if needed).
3) Sync InfantDistressClassification code to $PROJ/code/...
4) Either:
     sbatch slurm/af3_lab_score.slurm
   or run submit_on_sapelo2.sh after editing MAP/ADAPTER paths.
5) Download windows_scored.csv into this run folder and paste path into GUI Scores CSV.
6) Run Phase 1 in the GUI.
""",
            encoding="utf-8",
        )
        return (
            f"Sapelo2 bundle ready: {bundle}\n"
            f"{map_msg}\n"
            f"{len(am)} files listed in wav_list.txt\n"
            "See README.txt in that folder."
        )
    except Exception:
        return traceback.format_exc()


SCORE_HELP = """
### AF3 scoring

**This computer (GPU):** set **Adapter dir** to your `lora_adapter` folder (must contain `adapter_config.json`), then click **Run AF3 scoring**.  
Daylong files are slow on a laptop GPU — use **Limit files** / **Limit windows per file** to smoke-test first.

**Sapelo2 (recommended for full daylong runs):** click **Prepare Sapelo2 bundle**, follow `sapelo2_bundle/README.txt`, then paste the downloaded scores CSV above and continue to Phase 1.

Do **not** concatenate multi-day recordings. Multiple files per child are supported.
"""


def build_app():
    import gradio as gr

    default_run = str(REPO_ROOT / "lab_runs" / "my_first_run")
    default_scores = str(REPO_ROOT / "demo" / "sample_scores.csv")
    default_audio = ""
    default_adapter = ""

    with gr.Blocks(title="Preschooler Distress Pipeline") as demo:
        gr.Markdown(
            """
# Preschooler Distress Detection Pipeline
Fine-tuned **Audio Flamingo 3** → review clips → human notes → episodes.

Use both review strategies if needed. Keep multiple recordings per child as **separate files** (do not merge).
"""
        )

        with gr.Row():
            run_dir = gr.Textbox(label="Run folder", value=default_run, scale=2)
            scores_csv = gr.Textbox(label="Scores CSV (AF3 windows)", value=default_scores, scale=3)
        with gr.Row():
            audio_dir = gr.Textbox(label="Audio folder (WAVs)", value=default_audio, scale=3)
            audio_map_csv = gr.Textbox(label="Audio map CSV (optional)", value="", scale=2)
            btn_map = gr.Button("Build audio map from folder", scale=1)

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                threshold = gr.Number(label="Threshold (AF3 R≈0.70 → 0.32)", value=0.32)
                top_fraction = gr.Number(label="Cluster top fraction", value=0.50)
                write_clips = gr.Checkbox(label="Write review / episode WAV clips", value=True)
            with gr.Row():
                consecutive = gr.Checkbox(label="Review mode: consecutive peak windows", value=True)
                cluster_top50 = gr.Checkbox(label="Review mode: 30s clusters, top 50%", value=True)
            with gr.Row():
                episode_gap = gr.Number(label="Episode merge gap (sec)", value=300, precision=0)
                pre_buffer = gr.Number(label="Episode pre-buffer (sec)", value=60, precision=0)

        map_status = gr.Textbox(label="Audio map status", interactive=False)
        btn_map.click(build_audio_map, inputs=[audio_dir, run_dir], outputs=[audio_map_csv, map_status])

        with gr.Tab("1 · Score audio (AF3)"):
            gr.Markdown(SCORE_HELP)
            gpu_box = gr.Textbox(label="GPU status", value=gpu_status(), interactive=False)
            with gr.Row():
                adapter_dir = gr.Textbox(
                    label="LoRA adapter dir (folder with adapter_config.json)",
                    value=default_adapter,
                    scale=3,
                )
                base_only = gr.Checkbox(label="Base AF3 only (no LoRA — smoke test)", value=False)
            with gr.Row():
                limit_files = gr.Number(
                    label="Limit files (0 = all)", value=1, precision=0
                )
                limit_windows = gr.Number(
                    label="Limit windows per file (0 = all; try 20 to smoke-test)",
                    value=20,
                    precision=0,
                )
                resume = gr.Checkbox(label="Resume if scores CSV exists", value=True)
            with gr.Row():
                btn_score = gr.Button("Run AF3 scoring", variant="primary")
                btn_score_p1 = gr.Button("Score + Phase 1")
                btn_sapelo = gr.Button("Prepare Sapelo2 bundle")
            score_status = gr.Textbox(label="Scoring status", lines=10)
            sapelo_status = gr.Textbox(label="Sapelo2 bundle status", lines=6)

            def _score_only(run_dir, audio_dir, audio_map_csv, adapter_dir, base_only, limit_files, limit_windows, resume):
                msg, path = run_af3_score(
                    run_dir, audio_dir, audio_map_csv, adapter_dir, base_only, limit_files, limit_windows, resume
                )
                return msg, path if path else gr.update()

            def _score_then_p1(
                run_dir,
                audio_dir,
                audio_map_csv,
                adapter_dir,
                base_only,
                limit_files,
                limit_windows,
                resume,
                threshold,
                consecutive,
                cluster_top50,
                write_clips,
                top_fraction,
                episode_gap,
                pre_buffer,
            ):
                msg, path = run_af3_score(
                    run_dir, audio_dir, audio_map_csv, adapter_dir, base_only, limit_files, limit_windows, resume
                )
                if not path:
                    return msg, "", None, None, None
                p1_msg, pos, c, k = run_phase1(
                    run_dir,
                    path,
                    audio_dir,
                    audio_map_csv,
                    threshold,
                    consecutive,
                    cluster_top50,
                    write_clips,
                    top_fraction,
                    episode_gap,
                    pre_buffer,
                )
                return f"{msg}\n\n{p1_msg}", path, pos, c, k

            btn_score.click(
                _score_only,
                inputs=[run_dir, audio_dir, audio_map_csv, adapter_dir, base_only, limit_files, limit_windows, resume],
                outputs=[score_status, scores_csv],
            )
            btn_sapelo.click(
                prepare_sapelo2_bundle,
                inputs=[run_dir, audio_dir, audio_map_csv],
                outputs=[sapelo_status],
            )

        with gr.Tab("2 · Phase 1 — Review clips"):
            btn_p1 = gr.Button("Run Phase 1", variant="primary")
            p1_status = gr.Textbox(label="Status", lines=8)
            with gr.Row():
                pos_df = gr.Dataframe(label="Positive windows (preview)", interactive=False)
            with gr.Row():
                consec_df = gr.Dataframe(label="Consecutive segments (preview)", interactive=False)
                clust_df = gr.Dataframe(label="Cluster top-50% (preview)", interactive=False)
            btn_p1.click(
                run_phase1,
                inputs=[
                    run_dir,
                    scores_csv,
                    audio_dir,
                    audio_map_csv,
                    threshold,
                    consecutive,
                    cluster_top50,
                    write_clips,
                    top_fraction,
                    episode_gap,
                    pre_buffer,
                ],
                outputs=[p1_status, pos_df, consec_df, clust_df],
            )

        with gr.Tab("3 · Review notes"):
            gr.Markdown(
                "Mark **is_true_distress** as `1` (true cry) or `0` (false). "
                "Listen to clips in the run folder `output/clips_*`."
            )
            with gr.Row():
                notes_mode = gr.Radio(
                    choices=["consecutive", "cluster_top50"],
                    value="consecutive",
                    label="Which review strategy notes?",
                )
                reviewer = gr.Textbox(label="Reviewer name", value="")
                btn_load = gr.Button("Load notes")
                btn_save = gr.Button("Save notes", variant="primary")
            notes_status = gr.Textbox(label="Status", interactive=False)
            notes_df = gr.Dataframe(label="Review notes (editable)", interactive=True, wrap=True)
            btn_load.click(load_notes, inputs=[run_dir, notes_mode], outputs=[notes_df, notes_status])
            btn_save.click(
                save_notes,
                inputs=[run_dir, notes_mode, notes_df, reviewer],
                outputs=[notes_status],
            )

        with gr.Tab("4 · Phase 2 — Episodes"):
            p2_mode = gr.Radio(
                choices=["consecutive", "cluster_top50"],
                value="consecutive",
                label="Notes / strategy to use for episodes",
            )
            btn_p2 = gr.Button("Run Phase 2", variant="primary")
            p2_status = gr.Textbox(label="Status", lines=8)
            with gr.Row():
                secs_df = gr.Dataframe(label="Human-reviewed distress seconds (preview)", interactive=False)
                eps_df = gr.Dataframe(label="Episodes (preview)", interactive=False)
            btn_p2.click(
                run_phase2,
                inputs=[
                    run_dir,
                    p2_mode,
                    scores_csv,
                    audio_dir,
                    audio_map_csv,
                    threshold,
                    consecutive,
                    cluster_top50,
                    write_clips,
                    top_fraction,
                    episode_gap,
                    pre_buffer,
                ],
                outputs=[p2_status, secs_df, eps_df],
            )

        with gr.Tab("5 · Download clips"):
            btn_zip = gr.Button("Zip clip folders for download")
            zip_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                z_consec = gr.File(label="clips_consecutive.zip")
                z_clust = gr.File(label="clips_cluster_top50.zip")
                z_ep = gr.File(label="episode_clips.zip")
            btn_zip.click(
                make_download_zips,
                inputs=[run_dir],
                outputs=[z_consec, z_clust, z_ep, zip_status],
            )

        gr.Markdown(
            "*Cross-platform Gradio GUI. AF3 scoring runs here if a GPU + LoRA adapter "
            "are available; otherwise use the Sapelo2 bundle, then Phase 1–2 locally.*"
        )

        # Re-bind Score+Phase1 to include Phase 1 preview tables
        btn_score_p1.click(
            _score_then_p1,
            inputs=[
                run_dir,
                audio_dir,
                audio_map_csv,
                adapter_dir,
                base_only,
                limit_files,
                limit_windows,
                resume,
                threshold,
                consecutive,
                cluster_top50,
                write_clips,
                top_fraction,
                episode_gap,
                pre_buffer,
            ],
            outputs=[score_status, scores_csv, pos_df, consec_df, clust_df],
        )

    return demo


def main() -> None:
    demo = build_app()
    # allow_reloading / reuse port if previous GUI still up
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
