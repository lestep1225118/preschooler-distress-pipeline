# Scoring audio on your own GPU (Windows)

This lets you run the AF3 scoring step on your own computer instead of Sapelo2. It needs an NVIDIA graphics card with at least 12 GB of its own memory (an RTX 3090 has 24 GB).

You only do Parts 1 and 2 once.

---

## What you need

- The app already installed (`START_HERE.md`, steps 1–3)
- About **40 GB of free disk space**. The first run downloads the base AF3 model (~32 GB) into your user folder.
- The trained model folder (`lora_adapter`), shared with you via UGA OneDrive

---

## Part 1 — Install the GPU packages (one time, ~10 minutes)

Open PowerShell in the `preschooler-distress-pipeline` folder (right-click the folder → **Open in Terminal**) and run these two commands, one at a time:

```
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu130
py -3.12 -m pip install -r requirements-scoring.txt
```

The first one is a large download (~2.5 GB).

**Check it worked:**

```
py -3.12 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True NVIDIA GeForce RTX 3090`. If it prints `False`, see Troubleshooting below.

---

## Part 2 — Put the model folder in place (one time)

1. Download the model zip from the OneDrive link and unzip it somewhere easy, e.g. `C:\Users\<you>\Documents\af3_model\`
2. Open the unzipped folder and find the folder named **`lora_adapter`**. It must directly contain a file called `adapter_config.json`.
3. Copy that folder's path: click the address bar in File Explorer while inside `lora_adapter` and copy it. It will look like  
   `C:\Users\<you>\Documents\af3_model\lora_adapter`

---

## Part 3 — Quick 20-window test

1. Put **one** WAV recording in a folder by itself.
2. Start the app:
   ```
   py -3.12 -m lab_pipeline.gui_app
   ```
   and open **http://127.0.0.1:7860**
3. At the top, set **Audio folder (WAVs)** to that folder.
4. Open tab **1 · Score audio (AF3)**:
   - **GPU status** should say `GPU available: NVIDIA GeForce RTX 3090 (24.0 GB)`
   - Paste the `lora_adapter` path into **LoRA adapter dir**
   - Leave **Limit files = 1** and **Limit windows per file = 20**
   - Click **Run AF3 scoring**

**The first time, this takes a while** (often 30–60 minutes) because it downloads the base AF3 model. You can watch progress in the PowerShell window. Later runs skip the download.

**It worked if** the Scoring status box says `AF3 scoring complete … Wrote 20 windows`. Note how long the 20 windows took after the download finished; that gives you the speed for full runs (a 6-hour recording is 4,320 windows).

You can then go to tab **2 · Phase 1** and run it on these 20 windows to check the whole chain.

---

## Part 4 — Full runs

Same as the test, but set **Limit files = 0** and **Limit windows per file = 0** (0 means "all").

- Keep **Resume** checked. If the run stops (computer sleeps, window closed), clicking **Run AF3 scoring** again continues where it left off instead of starting over.
- Set Windows to **never sleep** while plugged in during long runs (Settings → System → Power).
- Use a **new Run folder** (top of the app) for each new batch of recordings, so results don't mix.
- Several WAVs in the audio folder are fine; each is scored separately.

When it finishes, the scores CSV is filled in automatically. Continue with Phase 1 → Review notes → Phase 2 as in the tutorial video.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` prints `False` | Re-run the first install command in Part 1. If `pip` says torch is "already satisfied", run `py -3.12 -m pip uninstall -y torch` first, then install again. |
| GPU status says "No CUDA GPU detected" | Same fix as above, then restart the app. |
| `AudioFlamingo3ForConditionalGeneration` not found / cannot import | `py -3.12 -m pip install -U "transformers>=5.0"` |
| `LoRA adapter folder not found` or `No adapter_config.json` | The path must point at the `lora_adapter` folder itself, not the folder above it. |
| `CUDA out of memory` | Close other programs using the GPU (games, video editors, other Python windows) and try again. |
| Error mentioning `bitsandbytes` | `py -3.12 -m pip install -U bitsandbytes` |
| Download stops partway | Click **Run AF3 scoring** again; it resumes the download. |

If something else goes wrong, copy the full text from the **Scoring status** box (and the last lines of the PowerShell window) into an email.
