# Preschooler Distress Detection Pipeline

Lab-facing package with **two apps**:

1. **AF3 distress screening** (`lab_pipeline`) — fine-tuned Audio Flamingo 3
2. **VTC conversational turns** (`vtc_pipeline`) — adult ↔ key-child turn counts from VTC speaker labels

- Cross-platform **GUI** (Gradio) — Windows / Mac / Linux
- Config / folder workflow for scripting
- **No participant audio or model weights** are included in this repository

**Read `START_HERE.md` first** if you are not a programmer.

## Features

### AF3 distress (matches lab workflow)

1. Read audio files (multi-file participants supported; do not concatenate)
2. AF3 scoring → 5 s windows with timestamps + probabilities
3. Two review strategies (consecutive / 30 s cluster top 50%)
4. Review clips + notes → human distress seconds → 5-min episodes (+ 1 min buffer)

### VTC conversational turns

1. VTC speaker labels on Sapelo2 (`rttm.csv`: FEM / MAL / KCHI / OCH)
2. Count adult ↔ KCHI turns (≤5 s gap)
3. Report totals + per 5-minute block (zeros included)

## Quick start

```bash
python -m pip install -r requirements.txt

# AF3 app → http://127.0.0.1:7860
python -m lab_pipeline.gui_app

# VTC turns app → http://127.0.0.1:7861
python -m vtc_pipeline.gui_app
```

AF3 demo scores: `demo/sample_scores.csv`  
VTC demo rttm: `vtc_pipeline/demo/sample_rttm.csv` (click **Run demo**)

## Scoring AF3 (GPU)

```bash
python -m pip install -r requirements-scoring.txt
python -m lab_pipeline.score_audio_folder \
  --audio-map-csv path/to/audio_map.csv \
  --adapter-dir path/to/lora_adapter \
  --out-csv lab_runs/my_run/output/windows_scored.csv \
  --resume
```

Sapelo2: see `slurm/` and the AF3 GUI “Prepare Sapelo2 bundle” button.  
VTC Sapelo steps: `vtc_pipeline/sapelo/README.md`.

## Layout

```text
START_HERE.md          # non-technical setup (both apps)
README.md              # this file
requirements.txt       # GUI + analysis deps
requirements-scoring.txt
demo/sample_scores.csv
lab_pipeline/          # AF3 app + GUI
vtc_pipeline/          # conversational turns app + GUI + Sapelo notes
slurm/                 # AF3 Sapelo2 job scripts
```

## Privacy

- Do not commit WAV files, annotations, or LoRA adapters with participant-linked training data.
- Share adapters separately via secure lab storage if needed.
