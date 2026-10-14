# START HERE — Preschooler Distress Detection Pipeline

This folder is a **standalone** lab package with two apps:

1. **AF3 distress screening** (`lab_pipeline`) — Audio Flamingo 3  
2. **VTC conversational turns** (`vtc_pipeline`) — adult ↔ key-child turn counts  

It does **not** include participant audio or large model weights (those stay private).

---

## 1. Install Python

Install **Python 3.12**. Windows installer (direct link to 3.12.10, the last 3.12 release with an installer):  
https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe  
(Check “Add python.exe to PATH” on the first screen.)

Avoid the newest Python release: some packages don't have ready-made installers for it yet, and `pip install` fails with “Microsoft Visual C++ 14.0 or greater is required”. If you have several Python versions, use `py -3.12` in place of `python` in the commands below.

## 2. Open a terminal in this folder

- Windows: open the folder in File Explorer → address bar → type `powershell` → Enter  
- Or: open Terminal / Command Prompt and `cd` into this folder

## 3. Install packages

```bash
python -m pip install -r requirements.txt
```

---

# App A — AF3 distress pipeline

```bash
python -m lab_pipeline.gui_app
```

Browser: **http://127.0.0.1:7860**

### Quick demo (no scoring)

1. Leave **Scores CSV** as `demo/sample_scores.csv`  
2. Leave **Audio folder** empty (clips will be skipped)  
3. Uncheck **Write review / episode WAV clips** in Settings  
4. Tab **2 · Phase 1** → **Run Phase 1**  
5. Tab **3 · Review notes** → Load → mark some rows `1` / `0` → Save  
6. Tab **4 · Phase 2** → Run Phase 2  

### Real scoring

Needs WAV folder + LoRA adapter + GPU (or Sapelo2). See `lab_pipeline/README.md`.

---

# App B — VTC conversational turns

```bash
python -m vtc_pipeline.gui_app
```

Browser: **http://127.0.0.1:7861** (AF3 stays on 7860)

### Quick demo

1. Leave the demo `rttm.csv` path as-is (or click **Run demo**)  
2. Click **Compute turns**  
3. Download the three CSVs: totals, by 5-min block, detail  

**Turn definition:** adult (FEM/MAL) and key child (KCHI) speech within 5 seconds of each other.  
Counts are reported per **5-minute block** (zeros included) and for the **full recording**.

### Real data

1. Run **VTC** speaker labeling on **Sapelo2** (Linux/GPU) — see `vtc_pipeline/sapelo/README.md`  
2. Download `rttm.csv`  
3. Point the VTC app at that file → Compute turns  

More detail: `vtc_pipeline/README.md`

---

## What each app produces

| App | Output |
|-----|--------|
| AF3 | Scored 5 s windows → review clips/notes → 5-min distress episodes |
| VTC turns | `conversational_turns_total.csv`, `conversational_turns_by_5min.csv`, `conversational_turns_detail.csv` |
