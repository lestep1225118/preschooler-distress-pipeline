# VTC conversational-turns pipeline

Takes **VTC** speaker segments (`rttm.csv` with FEM / MAL / KCHI / OCH) and computes
**conversational turns** plus **speech duration** per 5-minute block.

## Turn definition

Speech by one party followed by speech by the other within **5 seconds**
(gap from end of first segment to start of second).

## Turn pairs (per 5-min block + recording total)

| Column | Pair |
|--------|------|
| `turns_adult_kchi` | Adult (FEM or MAL) ↔ key child |
| `turns_FEM_kchi` / `turns_MAL_kchi` | Breakdown of adult ↔ KCHI |
| `turns_mal_fem` | Male adult ↔ female adult |
| `turns_kchi_och` | Key child ↔ other child |

## Speech duration (seconds per 5-min block + total)

| Column | Label |
|--------|-------|
| `speech_duration_FEM_s` | Adult female |
| `speech_duration_MAL_s` | Adult male |
| `speech_duration_KCHI_s` | Key child |
| `speech_duration_OCH_s` | Other child |

Segment time is clipped to each block boundary (overlap counted in each block).

## Outputs

| File | Description |
|------|-------------|
| `conversational_turns_total.csv` | Totals per recording |
| `conversational_turns_by_5min.csv` | Per 5-minute block (**zeros included**) |
| `conversational_turns_detail.csv` | One row per turn (`turn_type` column) |

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
