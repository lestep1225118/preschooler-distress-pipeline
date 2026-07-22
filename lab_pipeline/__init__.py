"""Shared helpers for the lab distress-detection pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


WINDOW_SEC_DEFAULT = 5
STRIDE_SEC = 5


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_scores(scores_path: str) -> pd.DataFrame:
    """Load one CSV or all CSVs under a directory."""
    p = Path(scores_path)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSVs in {p}")
        frames = [pd.read_csv(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        if not p.is_file():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)

    required = {"start_second", "pred_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scores CSV missing columns {missing}")

    if "end_second" not in df.columns:
        df["end_second"] = df["start_second"].astype(int) + WINDOW_SEC_DEFAULT

    if "file_id" not in df.columns:
        if "participant_id" in df.columns:
            df["file_id"] = df["participant_id"].astype(str)
        else:
            raise ValueError("Scores need file_id or participant_id")

    df["file_id"] = df["file_id"].astype(str)
    if "participant_id" not in df.columns:
        # Best-effort: leading digits of file_id
        df["participant_id"] = df["file_id"].str.extract(r"^(\d+)", expand=False)

    df["start_second"] = df["start_second"].astype(int)
    df["end_second"] = df["end_second"].astype(int)
    df["pred_score"] = pd.to_numeric(df["pred_score"], errors="coerce")
    return df


def resolve_audio_map(cfg: Dict[str, Any], file_ids: List[str]) -> Dict[str, str]:
    """Return file_id -> wav_path."""
    mapping: Dict[str, str] = {}
    audio_map_csv = cfg.get("audio_map_csv")
    if audio_map_csv:
        am = pd.read_csv(audio_map_csv)
        if "file_id" not in am.columns or "wav_path" not in am.columns:
            raise ValueError("audio_map_csv needs columns: file_id, wav_path")
        for _, row in am.iterrows():
            mapping[str(row["file_id"])] = str(row["wav_path"])
        return mapping

    audio_dir = cfg.get("audio_dir")
    if not audio_dir:
        return mapping

    template = cfg.get("audio_name_template") or "{file_id}_spliced_6hours.wav"
    root = Path(audio_dir)
    for fid in file_ids:
        candidates = [
            root / template.format(file_id=fid, participant_id=fid),
            root / f"{fid}.wav",
            root / f"{fid}_spliced_6hours.wav",
        ]
        # recursive fallback: exact stem match
        hit = next((c for c in candidates if c.is_file()), None)
        if hit is None:
            matches = list(root.rglob(f"{fid}.wav")) + list(
                root.rglob(f"{fid}_spliced_6hours.wav")
            )
            if matches:
                hit = matches[0]
        if hit is not None:
            mapping[fid] = str(hit)
    return mapping


def cluster_consecutive(starts: List[int], stride: int = STRIDE_SEC) -> List[List[int]]:
    if not starts:
        return []
    ordered = sorted(starts)
    clusters: List[List[int]] = [[ordered[0]]]
    for s in ordered[1:]:
        if s - clusters[-1][-1] == stride:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return clusters


def cluster_by_gap(starts: List[int], max_gap: int) -> List[List[int]]:
    if not starts:
        return []
    ordered = sorted(starts)
    clusters: List[List[int]] = [[ordered[0]]]
    for s in ordered[1:]:
        if s - clusters[-1][-1] <= max_gap:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return clusters


def span_from_starts(starts: List[int], window_sec: int) -> Tuple[int, int]:
    return int(starts[0]), int(starts[-1]) + window_sec


def positive_starts(df_file: pd.DataFrame, threshold: float) -> List[int]:
    sub = df_file[df_file["pred_score"].astype(float) >= threshold]
    return sorted(int(x) for x in sub["start_second"].tolist())


def build_segment_table(
    scores: pd.DataFrame,
    threshold: float,
    *,
    mode: str,
    window_sec: int,
    cluster_gap_sec: int = 30,
    top_fraction: Optional[float] = None,
) -> pd.DataFrame:
    """
    mode:
      - consecutive: merge adjacent positive 5 s windows
      - cluster: 30 s gap clustering; optionally keep top_fraction by rank
    """
    rows: List[dict] = []
    for file_id, sub in scores.groupby("file_id", sort=True):
        pid = sub["participant_id"].iloc[0] if "participant_id" in sub.columns else file_id
        starts = positive_starts(sub, threshold)
        if mode == "consecutive":
            groups = cluster_consecutive(starts)
        elif mode == "cluster":
            groups = cluster_by_gap(starts, cluster_gap_sec)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for starts_g in groups:
            wins = sub[sub["start_second"].isin(starts_g)]
            prob_max = float(wins["pred_score"].max())
            prob_mean = float(wins["pred_score"].mean())
            peak_idx = wins["pred_score"].idxmax()
            peak_start = int(wins.loc[peak_idx, "start_second"])
            peak_end = int(wins.loc[peak_idx, "end_second"])
            peak_score = float(wins.loc[peak_idx, "pred_score"])
            c_start, c_end = span_from_starts(starts_g, window_sec)
            sharp = 1.0 if (prob_mean > 0 and prob_max / prob_mean >= 1.15) else 0.0
            rank_score = prob_max + prob_mean + 0.15 * sharp
            rows.append(
                {
                    "participant_id": pid,
                    "file_id": str(file_id),
                    "mode": mode,
                    "segment_start": c_start,
                    "segment_end": c_end,
                    "segment_span_sec": c_end - c_start,
                    "n_windows": len(starts_g),
                    "prob_max": prob_max,
                    "prob_mean": prob_mean,
                    "rank_score": rank_score,
                    "peak_window_start": peak_start,
                    "peak_window_end": peak_end,
                    "peak_pred_score": peak_score,
                    "window_starts": ",".join(str(s) for s in starts_g),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("rank_score", ascending=False).reset_index(drop=True)
    out["global_rank"] = np.arange(1, len(out) + 1)
    if top_fraction is not None:
        n_keep = max(1, int(np.ceil(len(out) * float(top_fraction))))
        out["in_top_fraction"] = out["global_rank"] <= n_keep
        out = out[out["in_top_fraction"]].copy()
    else:
        out["in_top_fraction"] = True
    out["segment_id"] = np.arange(1, len(out) + 1)
    return out


def write_peak_clips(
    segments: pd.DataFrame,
    audio_map: Dict[str, str],
    clips_dir: Path,
) -> pd.DataFrame:
    """Cut peak 5 s windows; attach clip_path column."""
    import soundfile as sf

    from lab_pipeline.audio_io import read_wav_segment

    ensure_dir(clips_dir)
    clip_paths: List[Optional[str]] = []
    wav_paths: List[Optional[str]] = []
    notes: List[str] = []

    for _, row in segments.iterrows():
        fid = str(row["file_id"])
        wav = audio_map.get(fid)
        wav_paths.append(wav)
        if not wav or not os.path.isfile(wav):
            clip_paths.append(None)
            notes.append("missing_wav")
            continue
        start = int(row["peak_window_start"])
        end = int(row["peak_window_end"])
        try:
            audio, sr = read_wav_segment(wav, start, end)
            rel = f"{fid}_peak_{start:06d}.wav"
            out_path = clips_dir / rel
            sf.write(str(out_path), audio, sr)
            clip_paths.append(str(out_path))
            notes.append("ok")
        except Exception as e:  # noqa: BLE001
            clip_paths.append(None)
            notes.append(f"clip_error:{e}")

    out = segments.copy()
    out["wav_path"] = wav_paths
    out["clip_path"] = clip_paths
    out["clip_status"] = notes
    return out


def make_review_notes_template(segments: pd.DataFrame) -> pd.DataFrame:
    """Empty review sheet for RAs."""
    cols = [
        "segment_id",
        "participant_id",
        "file_id",
        "mode",
        "peak_window_start",
        "peak_window_end",
        "peak_pred_score",
        "segment_start",
        "segment_end",
        "clip_path",
    ]
    base = segments[[c for c in cols if c in segments.columns]].copy()
    base["is_true_distress"] = ""  # 1 / 0 / unsure
    base["reviewer"] = ""
    base["notes"] = ""
    return base


def merge_episodes(
    positive_starts: List[int],
    window_sec: int,
    gap_sec: int,
) -> List[Tuple[int, int]]:
    """Merge positive window starts into episodes if within gap_sec."""
    if not positive_starts:
        return []
    ordered = sorted(positive_starts)
    episodes: List[List[int]] = [[ordered[0]]]
    for s in ordered[1:]:
        prev_end = episodes[-1][-1] + window_sec
        if s - prev_end <= gap_sec:
            episodes[-1].append(s)
        else:
            episodes.append([s])
    return [(ep[0], ep[-1] + window_sec) for ep in episodes]
