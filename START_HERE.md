# START HERE — Preschooler Distress Detection Pipeline

This folder is a **standalone** app for screening preschooler distress in home audio using fine-tuned Audio Flamingo 3 (AF3).

It does **not** include participant audio or the large model adapter (those stay private).

---

## 1. Install Python

Install **Python 3.10+** from https://www.python.org/downloads/  
(On Windows, check “Add Python to PATH”.)

## 2. Open a terminal in this folder

- Windows: open the folder in File Explorer → address bar → type `powershell` → Enter  
- Or: open Terminal / Command Prompt and `cd` into this folder

## 3. Install the basic packages

```bash
python -m pip install -r requirements.txt
```

## 4. Open the app

```bash
python -m lab_pipeline.gui_app
```

A browser should open to **http://127.0.0.1:7860**

## 5. Try without scoring (demo)

1. Leave **Scores CSV** as `demo/sample_scores.csv`  
2. Leave **Audio folder** empty (clips will be skipped)  
3. Uncheck **Write review / episode WAV clips** in Settings  
4. Go to tab **2 · Phase 1** → click **Run Phase 1**  
5. Open tab **3 · Review notes** → Load notes → set some rows to `1` / `0` → Save  
6. Tab **4 · Phase 2** → Run Phase 2  

This proves the review → notes → episodes path works.

## 6. Score your own audio (real use)

You need:

1. A folder of **WAV** files (one file per recording — do **not** merge days)  
2. The fine-tuned **LoRA adapter** folder (contains `adapter_config.json`) — copy from Sapelo2 / Drive  
3. A **GPU** computer, **or** Sapelo2 for daylong files  

**On this computer (GPU + adapter):**  
- Set Audio folder → Build audio map  
- Set Adapter dir → Run AF3 scoring (start with Limit files = 1, Limit windows = 20)  
- Then Phase 1 → notes → Phase 2  

**On Sapelo2:**  
- Tab 1 → Prepare Sapelo2 bundle → follow `sapelo2_bundle/README.txt`  
- Also see `slurm/` job scripts  

For scoring dependencies on a GPU machine:

```bash
python -m pip install -r requirements-scoring.txt
```

---

## What the app produces

| Step | Output |
|------|--------|
| Score | `windows_scored.csv` — every 5 s window + probability |
| Phase 1 | Positive windows; consecutive + top-50% cluster reviews; 5 s clips; notes templates |
| Notes | You mark true/false distress |
| Phase 2 | Human distress-seconds file; 5-minute episodes; episode clips (+ 1 min buffer) |

More detail: `README.md` and `lab_pipeline/README.md`
