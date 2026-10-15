#!/bin/bash
#SBATCH --job-name=vtc2
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vtc_%j.out
#SBATCH --error=logs/vtc_%j.err

# Submit from /scratch/$USER/vtc (log paths above are relative to it):
#   cd /scratch/$USER/vtc && sbatch run_vtc_lab.sbatch
# Converts every audio file in incoming/ to mono 16 kHz WAV, runs VTC on all
# of them, and writes out/rttm.csv (one row per speech segment, uid = file name).

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ml FFmpeg/6.0-GCCcore-12.3.0

VTC_HOME=/scratch/$USER/vtc
IN_DIR=$VTC_HOME/incoming
RUN_DIR=$VTC_HOME/audio_run
OUT_DIR=$VTC_HOME/out
CODE=$VTC_HOME/code/VTC

rm -rf "$RUN_DIR" "$OUT_DIR"
mkdir -p "$RUN_DIR" "$OUT_DIR"

shopt -s nullglob nocaseglob
files=("$IN_DIR"/*.wav "$IN_DIR"/*.mp3 "$IN_DIR"/*.flac "$IN_DIR"/*.m4a)
if [ ${#files[@]} -eq 0 ]; then
  echo "No audio files found in $IN_DIR" >&2
  exit 1
fi

for f in "${files[@]}"; do
  name=$(basename "${f%.*}")
  echo "Converting $f"
  ffmpeg -nostdin -loglevel error -y -i "$f" -ac 1 -ar 16000 "$RUN_DIR/$name.wav"
done

cd "$CODE"
uv run scripts/infer.py \
  --wavs "$RUN_DIR" \
  --output "$OUT_DIR" \
  --device cuda

echo "Done. Results: $OUT_DIR/rttm.csv"
