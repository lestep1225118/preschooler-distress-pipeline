# Run VTC on Sapelo2 (speaker labels)

VTC 2.x needs Linux + FFmpeg + GPU. Use Sapelo2 for daylong / long clips.

> **New users: follow `GACRC_GUIDE.md` instead.** It works for any GACRC account and uses `run_vtc_lab.sbatch`. The notes below are the original developer setup (paths tied to one account).

## One-time setup

```bash
export PROJ=/scratch/ls80136/infant_distress
mkdir -p $PROJ/vtc/{code,audio,audio_run,out,logs,turns}
export PATH="$HOME/.local/bin:$PATH"

# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

cd $PROJ/vtc/code
git clone https://github.com/LAAC-LSCP/VTC.git
cd VTC
uv sync
# download VTC-2 checkpoint via Hugging Face snapshot_download if not present
```

## Prepare audio (mono, 16 kHz WAV)

```bash
module load FFmpeg/6.0-GCCcore-12.3.0
# example: 30-min clip from daylong file
ffmpeg -y -ss 1800 -t 1800 -i $PROJ/data/Ravindran/105_spliced_6hours.wav \
  -ac 1 -ar 16000 $PROJ/vtc/audio/105_sample_30min.wav

rm -f $PROJ/vtc/audio_run/*
cp $PROJ/vtc/audio/105_sample_30min.wav $PROJ/vtc/audio_run/
```

## Submit inference

```bash
sbatch $PROJ/vtc/run_vtc.sbatch   # or copy from this repo's sapelo/ folder
squeue -u $USER
# when done:
head $PROJ/vtc/out/rttm.csv
```

## Turn counts (on Sapelo or locally)

```bash
# on Sapelo (af3 env has pandas):
$PROJ/env/af3/bin/python -m vtc_pipeline.conversational_turns \
  --rttm $PROJ/vtc/out/rttm.csv \
  --out-dir $PROJ/vtc/turns/run1
```

Or download `rttm.csv` to your PC and use the Gradio GUI / CLI there.
