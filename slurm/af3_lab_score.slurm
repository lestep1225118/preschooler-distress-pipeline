#!/bin/bash
#SBATCH --job-name=af3_lab_score
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ls80136/infant_distress/logs/af3_lab_score_%j.out
#SBATCH --error=/scratch/ls80136/infant_distress/logs/af3_lab_score_%j.err

# Score a folder of WAVs with AF3 LoRA, then run local Phase 1 review outputs.
# Upload WAVs to $AUDIO_DIR first (one file per recording; do NOT concatenate).

set -euo pipefail

export SCRATCH=/scratch/ls80136
export PROJ=$SCRATCH/infant_distress
export CODE=$PROJ/code/InfantDistressClassification
export HF_HOME=$PROJ/hf_cache
export PY=$PROJ/env/af3/bin/python

AUDIO_DIR=${AUDIO_DIR:-$PROJ/lab_incoming/audio}
OUT_CSV=${OUT_CSV:-$PROJ/lab_incoming/scores/windows_scored.csv}
# Prefer a fixed production adapter when available; else one LOPO fold adapter.
ADAPTER=${ADAPTER:-$PROJ/models/fold_105/lora_adapter}
RUN_DIR=${RUN_DIR:-$PROJ/lab_incoming/runs/$(date +%Y%m%d_%H%M%S)}

mkdir -p "$(dirname "$OUT_CSV")" "$RUN_DIR" "$PROJ/logs"
cd "$CODE"

echo "AUDIO_DIR=$AUDIO_DIR"
echo "OUT_CSV=$OUT_CSV"
echo "ADAPTER=$ADAPTER"
echo "RUN_DIR=$RUN_DIR"

$PY -m lab_pipeline.score_audio_folder \
  --audio-dir "$AUDIO_DIR" \
  --adapter-dir "$ADAPTER" \
  --out-csv "$OUT_CSV" \
  --window-sec 5 \
  --stride 5 \
  --recursive \
  --resume

# Write a run config for Phase 1 on the same machine (CPU is fine after scoring)
CONFIG=$RUN_DIR/config.yaml
cat > "$CONFIG" <<EOF
run_dir: $RUN_DIR
scores_csv: $OUT_CSV
audio_map_csv: null
audio_dir: $AUDIO_DIR
audio_name_template: "{file_id}.wav"
threshold: 0.32
modes:
  consecutive: true
  cluster_top50: true
cluster_gap_sec: 30
top_fraction: 0.50
window_sec: 5
write_clips: true
episode_gap_sec: 300
episode_pre_buffer_sec: 60
EOF

$PY -m lab_pipeline.run_phase1_review --config "$CONFIG"

echo "Done. Scores: $OUT_CSV"
echo "Review outputs: $RUN_DIR/output"
