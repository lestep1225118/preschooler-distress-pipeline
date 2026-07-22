# Preschooler Distress Detection Pipeline

Lab-facing pipeline for **fine-tuned Audio Flamingo 3** distress screening on naturalistic home recordings.

- Cross-platform **GUI** (Gradio) — Windows / Mac / Linux  
- Config / folder workflow for scripting  
- **No participant audio or model weights** are included in this repository  

**Read `START_HERE.md` first** if you are not a programmer.

## Features (matches lab workflow)

1. Read audio files (multi-file participants supported; do not concatenate)  
2. AF3 scoring → 5 s windows with timestamps + probabilities  
3. Two review strategies:  
   - consecutive positive windows → peak 5 s window  
   - 30 s gap clusters → top 50% → peak 5 s window  
4. Cut 5 s review clips  
5. Review notes → human-reviewed distress seconds  
6. Merge within 5 minutes → episodes  
7. Episode clips with 1-minute pre-buffer  

## Quick start

```bash
python -m pip install -r requirements.txt
python -m lab_pipeline.gui_app
```

Demo scores (no audio needed for Phase 1–2 tables): `demo/sample_scores.csv`

## Scoring (GPU)

```bash
python -m pip install -r requirements-scoring.txt
python -m lab_pipeline.score_audio_folder \
  --audio-map-csv path/to/audio_map.csv \
  --adapter-dir path/to/lora_adapter \
  --out-csv lab_runs/my_run/output/windows_scored.csv \
  --resume
```

Sapelo2: see `slurm/` and the GUI “Prepare Sapelo2 bundle” button.

## Layout

```text
START_HERE.md          # non-technical setup
README.md              # this file
requirements.txt       # GUI + Phase 1–2
requirements-scoring.txt
demo/sample_scores.csv
lab_pipeline/          # app code + GUI
audio_flamingo3_lopo_eval_colab.py  # AF3 scoring backend
slurm/                 # Sapelo2 job scripts
```

## Privacy

- Do not commit WAV files, annotations, or LoRA adapters with participant-linked training data.  
- Share adapters separately via secure lab storage if needed.
