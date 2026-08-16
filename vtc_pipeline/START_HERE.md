# VTC conversational turns — quick start

## Install (once)

```bash
python -m pip install -r requirements.txt
```

## Open the app

```bash
python -m vtc_pipeline.gui_app
```

→ **http://127.0.0.1:7861**

## Demo (no Sapelo needed)

Click **Run demo**. You should see **25** adult↔KCHI turns for the sample file, with all six 5-minute blocks (including zeros at 0–300s and 600–900s).

## Your own recordings

1. On Sapelo2, run VTC → get `rttm.csv` (`vtc_pipeline/sapelo/README.md`)  
2. In the app: set path or upload `rttm.csv` → **Compute turns**  
3. Download the three CSV outputs  

## CLI

```bash
python -m vtc_pipeline.conversational_turns --rttm path/to/rttm.csv --out-dir vtc_runs/my_run
```
