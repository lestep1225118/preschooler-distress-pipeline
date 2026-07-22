"""
Score WAV files with fine-tuned Audio Flamingo 3 (LoRA).

Writes a windows CSV compatible with lab_pipeline Phase 1:
  participant_id, file_id, start_second, end_second, pred_score [, pred_distress, response]

Inputs (pick one):
  --audio-dir   folder of WAVs (optional --recursive / --file-glob-stem)
  --audio-map-csv  CSV with columns file_id, wav_path [, participant_id]

Multiple files per participant are scored separately (do NOT concatenate).

Requires GPU + AF3 deps. Example:
  python -m lab_pipeline.score_audio_folder \\
    --audio-map-csv audio_map.csv \\
    --adapter-dir /path/to/lora_adapter \\
    --out-csv windows_scored.csv \\
    --resume
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import soundfile as sf

from audio_flamingo3_lopo_eval_colab import (
    load_base_model,
    load_model,
    predict_window,
    score_window,
)


def wav_duration_sec(wav_path: str) -> float:
    with sf.SoundFile(wav_path, "r") as f:
        return float(len(f) / float(f.samplerate))


def participant_from_stem(stem: str) -> str:
    m = re.match(r"^(\d+)", stem)
    return m.group(1) if m else stem


def normalize_file_id(stem: str) -> str:
    if stem.endswith("_spliced_6hours"):
        return stem[: -len("_spliced_6hours")]
    return stem


def discover_wavs(audio_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in audio_dir.rglob("*.wav") if p.is_file())
    # Prefer daylong spliced files when present
    spliced = sorted(audio_dir.glob("*_spliced_6hours.wav"))
    if spliced:
        return spliced
    return sorted(p for p in audio_dir.glob("*.wav") if p.is_file())


def jobs_from_audio_map(map_csv: Path) -> List[Tuple[str, str, Path]]:
    """Return list of (participant_id, file_id, wav_path)."""
    df = pd.read_csv(map_csv)
    if "wav_path" not in df.columns:
        raise SystemExit("audio_map_csv needs wav_path column")
    jobs = []
    for _, row in df.iterrows():
        wav = Path(str(row["wav_path"]))
        if "file_id" in df.columns and pd.notna(row.get("file_id")):
            fid = str(row["file_id"])
        else:
            fid = normalize_file_id(wav.stem)
        if "participant_id" in df.columns and pd.notna(row.get("participant_id")):
            pid = str(row["participant_id"])
        else:
            pid = participant_from_stem(fid)
        jobs.append((pid, fid, wav))
    return jobs


def jobs_from_dir(
    audio_dir: Path,
    recursive: bool,
    file_glob_stem: Optional[str],
    limit_files: Optional[int],
) -> List[Tuple[str, str, Path]]:
    wavs = discover_wavs(audio_dir, recursive)
    if file_glob_stem:
        rx = re.compile(file_glob_stem)
        wavs = [w for w in wavs if rx.search(w.stem)]
    if limit_files:
        wavs = wavs[:limit_files]
    jobs = []
    for wav in wavs:
        fid = normalize_file_id(wav.stem)
        pid = participant_from_stem(fid)
        jobs.append((pid, fid, wav))
    return jobs


def load_done(out_csv: Path) -> set:
    if not out_csv.is_file():
        return set()
    df = pd.read_csv(out_csv)
    if df.empty or "file_id" not in df.columns:
        return set()
    return {(str(r.file_id), int(r.start_second)) for r in df.itertuples(index=False)}


def append_rows(out_csv: Path, rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not out_csv.is_file()
    df.to_csv(out_csv, mode="a", header=header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="AF3 folder scorer for lab pipeline")
    parser.add_argument("--audio-dir", default=None, help="Folder of WAV files (not merged)")
    parser.add_argument("--audio-map-csv", default=None, help="CSV: file_id,wav_path[,participant_id]")
    parser.add_argument("--adapter-dir", default=None, help="Path to lora_adapter")
    parser.add_argument("--base-only", action="store_true", help="Score with base AF3 (no LoRA)")
    parser.add_argument("--out-csv", required=True, help="Output windows CSV path")
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--recursive", action="store_true", help="Find WAVs under subfolders")
    parser.add_argument("--limit-windows-per-file", type=int, default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-scored windows")
    parser.add_argument(
        "--full-predict",
        action="store_true",
        help="Also decode Yes/No text (slower). Default: P(Yes) logits only.",
    )
    parser.add_argument(
        "--file-glob-stem",
        default=None,
        help="Optional regex; only score stems matching this pattern",
    )
    args = parser.parse_args()

    if not args.audio_map_csv and not args.audio_dir:
        raise SystemExit("Pass --audio-dir and/or --audio-map-csv")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_csv.parent / "tmp_clips"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.audio_map_csv:
        jobs = jobs_from_audio_map(Path(args.audio_map_csv))
        if args.limit_files:
            jobs = jobs[: args.limit_files]
    else:
        audio_dir = Path(args.audio_dir)
        if not audio_dir.is_dir():
            raise SystemExit(f"audio-dir not found: {audio_dir}")
        jobs = jobs_from_dir(audio_dir, args.recursive, args.file_glob_stem, args.limit_files)

    if not jobs:
        raise SystemExit("No WAV files to score")

    if args.base_only:
        processor, model = load_base_model()
        print("Loaded base AF3 (no LoRA)", flush=True)
    else:
        if not args.adapter_dir:
            raise SystemExit("Pass --adapter-dir or --base-only")
        processor, model = load_model(args.adapter_dir)
        print(f"Loaded LoRA adapter: {args.adapter_dir}", flush=True)

    done = load_done(out_csv) if args.resume else set()
    if done:
        print(f"Resume: {len(done)} windows already scored", flush=True)

    print(f"Scoring {len(jobs)} files -> {out_csv}", flush=True)
    buffer: List[dict] = []
    n_scored = 0

    for wi, (pid, file_id, wav) in enumerate(jobs):
        if not wav.is_file():
            print(f"SKIP missing: {wav}", flush=True)
            continue
        try:
            dur = wav_duration_sec(str(wav))
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {wav.name}: cannot read ({e})", flush=True)
            continue

        max_start = int(dur) - args.window_sec
        if max_start < 0:
            print(f"SKIP {wav.name}: shorter than {args.window_sec}s", flush=True)
            continue
        starts = list(range(0, max_start + 1, args.stride))
        if args.limit_windows_per_file:
            starts = starts[: args.limit_windows_per_file]
        starts = [s for s in starts if (file_id, s) not in done]

        print(
            f"[{wi+1}/{len(jobs)}] {wav.name} (file_id={file_id}): "
            f"{len(starts)} windows (dur={dur:.0f}s)",
            flush=True,
        )

        for i, start in enumerate(starts):
            end = start + args.window_sec
            response = None
            pred = None
            score = None
            try:
                if args.full_predict:
                    response, pred, score = predict_window(
                        processor, model, str(wav), start, str(tmp_dir)
                    )
                else:
                    score = score_window(processor, model, str(wav), start, str(tmp_dir))
                    if score is not None:
                        pred = int(score >= 0.5)
            except Exception as e:  # noqa: BLE001
                response = f"ERROR: {e}"
                score = None
                pred = None

            row = {
                "participant_id": pid,
                "file_id": file_id,
                "start_second": start,
                "end_second": end,
                "pred_score": score,
                "pred_distress": pred,
            }
            if args.full_predict or response is not None:
                row["response"] = response
            buffer.append(row)
            n_scored += 1

            if len(buffer) >= 25:
                append_rows(out_csv, buffer)
                buffer = []
                print(f"  checkpoint {n_scored} windows", flush=True)

        if buffer:
            append_rows(out_csv, buffer)
            buffer = []

    print(f"Done. Wrote {n_scored} new windows to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
