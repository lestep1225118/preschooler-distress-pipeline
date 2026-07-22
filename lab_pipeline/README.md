# Lab distress-detection pipeline (folder + config)

Full workflow for fine-tuned **Audio Flamingo 3** distress screening.

```text
Audio WAVs (per recording, not merged)
        |
        v
 [Sapelo2 GPU]  score_audio_folder  (+ optional array job)
        |
        v
 windows_scored.csv + positive_windows.csv
        |
        v
 [local CPU]  Phase 1: both review strategies + 5 s peak clips + notes templates
        |
        v
 RA fills review_notes_*.csv  (is_true_distress)
        |
        v
 [local CPU]  Phase 2: human distress-seconds timeline → 5-min episodes → buffered clips
```

---

## GUI (dream version)

Cross-platform Gradio app (Windows / Mac / Linux via browser):

```bash
python -m pip install gradio
python -m lab_pipeline.gui_app
# or: lab_pipeline/launch_gui.ps1
```

Opens http://127.0.0.1:7860 with tabs for:
1. **Score audio (AF3)** — run scoring on this GPU (needs LoRA adapter path) or prepare a Sapelo2 bundle
2. Phase 1 review clips
3. Review notes
4. Phase 2 episodes
5. Zip downloads

For local scoring, copy a `lora_adapter` folder (with `adapter_config.json`) onto this machine and paste its path. Use limit files/windows to smoke-test before full daylong runs.

---

## Quick start (this machine)

```bash
python -m lab_pipeline.run_all --config lab_pipeline/config.yaml
```

Or double-click / run: `lab_pipeline/run_phase1.ps1`

Outputs under `lab_runs/full_pipeline_run/output/`:

| Output | Description |
|--------|-------------|
| `windows_scored.csv` | Every 5 s epoch + `pred_score` |
| `positive_windows.csv` | Epochs ≥ threshold (timestamps + probability) |
| `review_consecutive.csv` | Adjacent positives merged; peak 5 s window |
| `review_cluster_top50.csv` | 30 s clusters, top 50%, peak window |
| `review_notes_*.csv` | RA template |
| `clips_*` | 5 s review WAVs |

After RAs fill notes (`is_true_distress` = 1/0):

```bash
python -m lab_pipeline.run_all --config lab_pipeline/config.yaml --skip-phase1 --phase2 ^
  --review-notes lab_runs/full_pipeline_run/output/review_notes_consecutive.csv
```

Or: `lab_pipeline/run_phase2.ps1` (edit `$NOTES` path first).

Phase 2 writes:

| Output | Description |
|--------|-------------|
| `human_reviewed_distress_windows.csv` | Segments marked true |
| `human_reviewed_distress_seconds.csv` | Expanded second-level timeline |
| `episodes.csv` | 5-minute-gap episodes |
| `episode_clips/` | Episode audio + 1-minute pre-buffer |

---

## Sapelo2 — score new audio

### Production LoRA adapter

LOPO folds are for paper evaluation. For **new lab recordings**, use one designated adapter on Sapelo2:

```text
$PROJ/models/fold_105/lora_adapter
```

(default in the Slurm scripts). This is a stand-in until a final “train-on-all” production adapter is saved; swap `ADAPTER=` when that exists. Do **not** ship participant audio with the adapter.

### Single job (small batches)

1. Upload WAVs to `$PROJ/lab_incoming/audio` (one file per recording; do not concatenate).
2. Sync this repo to `$PROJ/code/InfantDistressClassification`.
3. Submit:

```bash
sbatch slurm/af3_lab_score.slurm
# overrides:
AUDIO_DIR=... ADAPTER=... sbatch slurm/af3_lab_score.slurm
```

### Parallel array (many files)

```bash
find $PROJ/lab_incoming/audio -name '*.wav' | sort > $PROJ/lab_incoming/wav_list.txt
N=$(wc -l < $PROJ/lab_incoming/wav_list.txt)
sbatch --array=0-$((N-1))%8 slurm/af3_lab_score_array.slurm

# after all tasks complete:
python -m lab_pipeline.merge_score_shards \
  --shards-dir $PROJ/lab_incoming/scores/shards \
  --out-csv $PROJ/lab_incoming/scores/windows_scored.csv
```

Then point `scores_csv` / `audio_dir` in config and run Phase 1 (CPU is fine).

---

## Multi-file participants

Do **not** merge day/session recordings. Auto-build a map:

```bash
python -m lab_pipeline.build_audio_map --audio-dir D:/lab_audio --out-csv audio_map.csv --recursive
```

Set `audio_map_csv` in `config.yaml`. Scores use the same `file_id` (filename stem).

---

## Review modes (both included)

1. **consecutive** — merge adjacent positive 5 s windows; review highest-probability window.
2. **cluster_top50** — 30 s gap clusters; top 50% by rank; peak 5 s window.

Default threshold `0.32` ≈ AF3 pooled recall 0.70.

---

## Dependencies

```bash
pip install pandas numpy pyyaml soundfile gradio
# Scoring (GPU / Sapelo2): use the existing AF3 conda env
```
