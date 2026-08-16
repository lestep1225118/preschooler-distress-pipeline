# VTC conversational-turns pipeline

Takes **VTC** speaker segments (`rttm.csv` with FEM / MAL / KCHI / OCH) and counts
**adult ↔ key-child conversational turns**.

## Definition

A turn = speech by one party (adult **FEM** or **MAL**, or key child **KCHI**)
followed by speech by the other within **5 seconds**
(gap from end of first segment to start of second). **OCH** is ignored.

## Outputs

| File | Description |
|------|-------------|
| `conversational_turns_total.csv` | Totals per recording (adult–KCHI, FEM–KCHI, MAL–KCHI) |
| `conversational_turns_by_5min.csv` | Same counts per 5-minute block (**zeros included**) |
| `conversational_turns_detail.csv` | One row per turn |

## GUI (local)

```bash
python -m pip install pandas gradio
python -m vtc_pipeline.gui_app
```

Opens http://127.0.0.1:7861 — point at an `rttm.csv`, click **Compute turns**.

Demo file: `vtc_pipeline/demo/sample_rttm.csv`

## CLI

```bash
python -m vtc_pipeline.conversational_turns \
  --rttm path/to/rttm.csv \
  --out-dir vtc_runs/my_run
```

## VTC speaker labeling (Sapelo2)

Windows cannot run VTC cleanly. Use Sapelo2: see `vtc_pipeline/sapelo/README.md`.
Then feed the resulting `rttm.csv` into this pipeline.
