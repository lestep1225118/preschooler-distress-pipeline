#!/bin/bash
#SBATCH --job-name=af3_lab_arr
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-99%8
#SBATCH --output=/scratch/ls80136/infant_distress/logs/af3_lab_arr_%A_%a.out
#SBATCH --error=/scratch/ls80136/infant_distress/logs/af3_lab_arr_%A_%a.err

# Parallel AF3 scoring: one WAV per array task.
# 1) Upload WAVs to $AUDIO_DIR
# 2) Build file list, then submit (adjust --array upper bound to n_files-1):
#      find $AUDIO_DIR -name '*.wav' | sort > $PROJ/lab_incoming/wav_list.txt
#      N=$(wc -l < $PROJ/lab_incoming/wav_list.txt)
#      sbatch --array=0-$((N-1))%8 slurm/af3_lab_score_array.slurm
# 3) After all tasks finish, merge + Phase 1:
#      python -m lab_pipeline.merge_score_shards --shards-dir $SCORE_DIR --out-csv ...
#      python -m lab_pipeline.run_phase1_review --config ...

set -euo pipefail

export SCRATCH=/scratch/ls80136
export PROJ=$SCRATCH/infant_distress
export CODE=$PROJ/code/InfantDistressClassification
export HF_HOME=$PROJ/hf_cache
export PY=$PROJ/env/af3/bin/python

AUDIO_DIR=${AUDIO_DIR:-$PROJ/lab_incoming/audio}
WAV_LIST=${WAV_LIST:-$PROJ/lab_incoming/wav_list.txt}
SCORE_DIR=${SCORE_DIR:-$PROJ/lab_incoming/scores/shards}
ADAPTER=${ADAPTER:-$PROJ/models/fold_105/lora_adapter}

mkdir -p "$SCORE_DIR" "$PROJ/logs"
cd "$CODE"

if [[ ! -f "$WAV_LIST" ]]; then
  find "$AUDIO_DIR" -type f -name '*.wav' | sort > "$WAV_LIST"
fi

N=$(wc -l < "$WAV_LIST" | tr -d ' ')
IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= N )); then
  echo "Index $IDX >= n_files $N — nothing to do"
  exit 0
fi

WAV=$(sed -n "$((IDX+1))p" "$WAV_LIST")
STEM=$(basename "$WAV" .wav)
OUT_CSV="$SCORE_DIR/${STEM}.csv"
TMP_DIR=$(mktemp -d)
ln -sf "$WAV" "$TMP_DIR/${STEM}.wav"

echo "Task $IDX / $N"
echo "WAV=$WAV"
echo "OUT=$OUT_CSV"
echo "ADAPTER=$ADAPTER"

$PY -m lab_pipeline.score_audio_folder \
  --audio-dir "$TMP_DIR" \
  --adapter-dir "$ADAPTER" \
  --out-csv "$OUT_CSV" \
  --window-sec 5 \
  --stride 5 \
  --resume

rm -rf "$TMP_DIR"
echo "Done $STEM"
