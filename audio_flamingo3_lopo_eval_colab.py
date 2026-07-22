"""
Colab: evaluate fine-tuned AF3 on held-out long file (LOPO test participant).

Scores 5 s windows every --stride seconds on the test participant's 6-hour wav,
using the same yes/no prompt as training.

Example:
  !python /content/audio_flamingo3_lopo_eval_colab.py \\
    --adapter-dir /content/af3_finetune_fold_105/lora_adapter \\
    --test-pid 105 \\
    --long-wav /content/Ravindran/105_spliced_6hours.wav \\
    --gt-csv /content/textgrid_csvs_recode/ID105_cryannotations_SK_finished.csv \\
    --out-dir /content/af3_eval_fold_105 \\
    --stride 5 --limit-windows 500
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

SCRIPT_VERSION = "2026-06-25"
MODEL_ID = "nvidia/audio-flamingo-3-hf"
PROMPT = (
    "Listen to this 5-second clip from a home recording of an infant. "
    "Does the clip contain an infant crying or in clear distress? "
    "Answer with exactly one word: Yes or No."
)


def load_gt_per_second(gt_csv_path: str) -> np.ndarray:
    df = pd.read_csv(gt_csv_path)
    col = df["crying"] if "crying" in df.columns else df.iloc[:, 2]
    labels = pd.to_numeric(col, errors="coerce").fillna(0).astype(float).to_numpy()
    return (labels > 0).astype(int)


def read_wav_segment(wav_path: str, start_second: int, end_second: int):
    import soundfile as sf

    with sf.SoundFile(wav_path, "r") as f:
        sr = int(f.samplerate)
        start_frame = int(start_second * sr)
        frames = int((end_second - start_second) * sr)
        f.seek(start_frame)
        audio = f.read(frames, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def parse_yes_no(text: str) -> Optional[int]:
    if not text or text.startswith("ERROR:"):
        return None
    t = text.lower().strip()
    has_yes = bool(re.search(r"\byes\b", t))
    has_no = bool(re.search(r"\bno\b", t))
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    if has_yes and has_no:
        return 1 if re.search(r"\byes\b", t).start() < re.search(r"\bno\b", t).start() else 0
    return None


def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def load_model(adapter_dir: str):
    adapter_dir = os.path.abspath(adapter_dir)
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"Adapter folder not found: {adapter_dir}\n"
            "Copy from Drive first, e.g.:\n"
            '  !cp -r "$DRIVE/af3_finetune_fold_105" /content/'
        )
    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"No adapter_config.json in {adapter_dir} — is this the lora_adapter folder?"
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    # Processor may be saved with adapter; otherwise load base model processor.
    proc_path = adapter_dir if os.path.isfile(
        os.path.join(adapter_dir, "processor_config.json")
    ) else MODEL_ID
    processor = AutoProcessor.from_pretrained(proc_path)
    base = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return processor, model


def yes_no_token_ids(processor) -> Tuple[Optional[int], Optional[int]]:
    tok = processor.tokenizer
    yes_ids = tok.encode("Yes", add_special_tokens=False)
    no_ids = tok.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0] if yes_ids else None
    no_id = no_ids[0] if no_ids else None
    return yes_id, no_id


def score_yes_probability(processor, model, inputs, prompt_len: int) -> Optional[float]:
    """P(Yes) from first generated token logits (Yes vs No)."""
    yes_id, no_id = yes_no_token_ids(processor)
    if yes_id is None or no_id is None:
        return None
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    logits = gen.scores[0][0]
    pair = torch.stack([logits[yes_id], logits[no_id]]).float()
    return float(torch.softmax(pair, dim=0)[0].item())


def score_window(
    processor,
    model,
    wav_path: str,
    start: int,
    tmp_dir: str,
) -> Optional[float]:
    """P(Yes) only — one short generate; faster than full predict_window."""
    import soundfile as sf

    end = start + 5
    audio, sr = read_wav_segment(wav_path, start, end)
    clip_path = os.path.join(tmp_dir, f"s{start}.wav")
    sf.write(clip_path, audio, sr)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "audio", "path": clip_path},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = inputs.to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    try:
        return score_yes_probability(processor, model, inputs, prompt_len)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def predict_window(
    processor,
    model,
    wav_path: str,
    start: int,
    tmp_dir: str,
) -> Tuple[str, Optional[int], Optional[float]]:
    import soundfile as sf

    end = start + 5
    audio, sr = read_wav_segment(wav_path, start, end)
    clip_path = os.path.join(tmp_dir, f"s{start}.wav")
    sf.write(clip_path, audio, sr)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "audio", "path": clip_path},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = inputs.to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    try:
        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        texts = processor.batch_decode(
            gen_ids[:, prompt_len:], skip_special_tokens=True
        )
        response = str(texts[0] if texts else "").strip()
        pred = parse_yes_no(response)
        score = score_yes_probability(processor, model, inputs, prompt_len)
        if score is None and pred is not None:
            score = float(pred)
        return response, pred, score
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_partial_predictions(out_csv: str) -> pd.DataFrame:
    if os.path.isfile(out_csv):
        return pd.read_csv(out_csv)
    return pd.DataFrame()


def write_summary(df: pd.DataFrame, args, summary_path: str) -> str:
    valid = df.dropna(subset=["pred_distress"]).copy()
    lines = [f"Test pid: {args.test_pid}", f"Windows: {len(df)}", ""]
    n_unparsed = int(df["pred_distress"].isna().sum())
    if n_unparsed:
        lines.append(f"Unparsed: {n_unparsed}/{len(df)}")
        lines.append("Sample unparsed responses:")
        for r in df.loc[df["pred_distress"].isna(), "response"].head(5):
            lines.append(f"  {repr(r)}")
        lines.append("")
    if len(valid):
        valid["pred_distress"] = valid["pred_distress"].astype(int)
        y_true = valid["gt_distress"].to_numpy()
        y_pred = valid["pred_distress"].to_numpy()
        labels = [0, 1]
        lines.extend(
            [
                f"Parsed: {len(valid)}/{len(df)}",
                f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}",
                f"Precision: {precision_score(y_true, y_pred, zero_division=0, labels=labels, average='binary', pos_label=1):.4f}",
                f"Recall:    {recall_score(y_true, y_pred, zero_division=0, labels=labels, average='binary', pos_label=1):.4f}",
                f"F1:        {f1_score(y_true, y_pred, zero_division=0, labels=labels, average='binary', pos_label=1):.4f}",
            ]
        )
        if "pred_score" in valid.columns and valid["pred_score"].notna().any():
            scores = valid["pred_score"].astype(float).to_numpy()
            try:
                lines.append(f"ROC-AUC:   {roc_auc_score(y_true, scores):.4f}")
                lines.append(f"PR-AUC:    {average_precision_score(y_true, scores):.4f}")
            except ValueError as exc:
                lines.append(f"AUC skipped: {exc}")
        lines.extend(
            [
                f"Confusion matrix (rows=true, cols=pred):\n{confusion_matrix(y_true, y_pred, labels=labels)}",
                "",
            ]
        )
        try:
            lines.append(
                classification_report(
                    y_true, y_pred, labels=labels, target_names=["no_distress", "distress"], zero_division=0
                )
            )
        except ValueError as exc:
            lines.append(f"classification_report skipped: {exc}")
            lines.append(f"pred_yes_rate={y_pred.mean():.3f}")
            lines.append(f"gt_yes_rate={y_true.mean():.3f}")
    summary = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default=None, help="Path to lora_adapter (omit with --base-only)")
    parser.add_argument("--base-only", action="store_true", help="Evaluate base AF3 without LoRA adapter")
    parser.add_argument("--test-pid", type=int, required=True)
    parser.add_argument("--long-wav", required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing lopo_eval_predictions.csv in --out-dir (skips done windows)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "tmp_clips")
    os.makedirs(tmp_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "lopo_eval_predictions.csv")
    summary_path = os.path.join(args.out_dir, "lopo_eval_summary.txt")

    gt = load_gt_per_second(args.gt_csv)
    if args.base_only:
        print("Mode: base model only (no adapter)", flush=True)
        processor, model = load_base_model()
    else:
        if not args.adapter_dir:
            parser.error("Pass --adapter-dir or use --base-only")
        processor, model = load_model(args.adapter_dir)

    rows = []
    starts = list(range(0, max(0, len(gt) - 5), args.stride))
    if args.limit_windows:
        starts = starts[: args.limit_windows]

    done_starts: set[int] = set()
    if args.resume:
        partial = load_partial_predictions(out_csv)
        if len(partial):
            rows = partial.to_dict("records")
            done_starts = set(int(x) for x in partial["start_second"].tolist())
            print(f"Resume: {len(done_starts)} windows already in {out_csv}", flush=True)
    starts = [s for s in starts if s not in done_starts]

    print(f"Script version: {SCRIPT_VERSION}", flush=True)
    if args.base_only:
        print("Model: base AF3 (no LoRA adapter)", flush=True)
    else:
        print(f"Adapter: {os.path.abspath(args.adapter_dir)}", flush=True)
    print(f"Evaluating {len(starts)} windows on pid {args.test_pid}", flush=True)

    n_prev = len(rows)
    for i, start in enumerate(starts):
        end = start + 5
        gt_label = int(np.mean(gt[start:end]) >= 0.5)
        try:
            response, pred, score = predict_window(processor, model, args.long_wav, start, tmp_dir)
        except Exception as exc:
            response, pred, score = f"ERROR: {exc}", None, None
        rows.append(
            {
                "participant_id": args.test_pid,
                "start_second": start,
                "end_second": end,
                "gt_distress": gt_label,
                "pred_distress": pred,
                "pred_score": score,
                "response": response,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {n_prev + i + 1}/{n_prev + len(starts)}", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = write_summary(df, args, summary_path)
    print(summary)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
