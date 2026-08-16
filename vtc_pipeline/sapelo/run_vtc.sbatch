#!/bin/bash
#SBATCH --job-name=vtc2
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ls80136/infant_distress/vtc/logs/vtc_%j.out
#SBATCH --error=/scratch/ls80136/infant_distress/vtc/logs/vtc_%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ml FFmpeg/6.0-GCCcore-12.3.0

# Put mono 16 kHz WAVs in audio_run/ (one file or many).
WAV_DIR=/scratch/ls80136/infant_distress/vtc/audio_run
OUT_DIR=/scratch/ls80136/infant_distress/vtc/out
CODE=/scratch/ls80136/infant_distress/vtc/code/VTC

cd "$CODE"
uv run scripts/infer.py \
  --wavs "$WAV_DIR" \
  --output "$OUT_DIR" \
  --device cuda
