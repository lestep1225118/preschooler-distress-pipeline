"""
Gradio GUI for VTC → adult ↔ key-child conversational turn counts.

Launch:
  python -m vtc_pipeline.gui_app
  # → http://127.0.0.1:7861
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from vtc_pipeline.conversational_turns import BLOCK_SEC, TURN_GAP_SEC, compute_turns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_RTTM = REPO_ROOT / "vtc_pipeline" / "demo" / "sample_rttm.csv"
DEFAULT_OUT = REPO_ROOT / "vtc_runs" / "gui_run"
SAPELO_README = REPO_ROOT / "vtc_pipeline" / "sapelo" / "README.md"


def _resolve_rttm(rttm_path: str, upload_path: str | None) -> Path:
    if upload_path:
        p = Path(upload_path)
        if p.is_file():
            return p
    if rttm_path and rttm_path.strip():
        p = Path(rttm_path.strip())
        if p.is_file():
            return p
    if DEMO_RTTM.is_file():
        return DEMO_RTTM
    raise FileNotFoundError("No rttm.csv found. Set a path, upload a file, or use the demo.")


def _run(
    rttm_path: str,
    upload_file: str | None,
    out_dir: str,
    gap_sec: float,
    block_sec: int,
) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, str, str, str]:
    try:
        rttm = _resolve_rttm(rttm_path, upload_file)
        out = Path(out_dir.strip()) if out_dir and out_dir.strip() else DEFAULT_OUT
        out.mkdir(parents=True, exist_ok=True)

        detail, blocks, total = compute_turns(
            rttm, out, gap_sec=float(gap_sec), block_sec=int(block_sec)
        )
        n_adult = int(total["turns_adult_kchi"].sum()) if not total.empty else 0
        msg = (
            f"Done.\n"
            f"Input: {rttm}\n"
            f"Output folder: {out}\n"
            f"Turn gap ≤ {gap_sec}s · block size = {block_sec}s\n"
            f"Detail turns: {len(detail)}\n"
            f"Total adult↔KCHI turns (all files): {n_adult}\n"
            f"Empty 5-min blocks are included as 0."
        )
        return (
            msg,
            total if not total.empty else None,
            blocks if not blocks.empty else None,
            str(out / "conversational_turns_total.csv"),
            str(out / "conversational_turns_by_5min.csv"),
            str(out / "conversational_turns_detail.csv"),
        )
    except Exception as e:
        return f"Error: {e}", None, None, "", "", ""


def _run_demo(out_dir: str, gap_sec: float, block_sec: int):
    return _run(str(DEMO_RTTM), None, out_dir, gap_sec, block_sec)


def _sapelo_text() -> str:
    if SAPELO_README.is_file():
        return SAPELO_README.read_text(encoding="utf-8")
    return "See vtc_pipeline/sapelo/README.md"


def build() -> gr.Blocks:
    with gr.Blocks(title="VTC conversational turns") as demo:
        gr.Markdown(
            """
# VTC → conversational turns

Counts **adult (FEM / MAL) ↔ key child (KCHI)** turns from a VTC `rttm.csv`.

| Setting | Default |
|---------|---------|
| Turn gap | 5 seconds (end of one → start of other) |
| Blocks | 5 minutes (empty blocks kept as **0**) |
| Ignored | OCH (other child) |

**Workflow:** run VTC on Sapelo2 → download `rttm.csv` → compute turns here (or on Sapelo).
"""
        )
        with gr.Tabs():
            with gr.Tab("1 · Compute turns"):
                with gr.Row():
                    rttm = gr.Textbox(
                        label="RTTM CSV path",
                        value=str(DEMO_RTTM) if DEMO_RTTM.is_file() else "",
                        info="Columns: uid, start_time_s, duration_s, label",
                    )
                    upload = gr.File(
                        label="Or upload rttm.csv",
                        file_types=[".csv"],
                        type="filepath",
                    )
                out_dir = gr.Textbox(label="Output folder", value=str(DEFAULT_OUT))
                with gr.Row():
                    gap = gr.Number(label="Turn gap (seconds)", value=TURN_GAP_SEC)
                    block = gr.Number(label="Block size (seconds)", value=BLOCK_SEC)
                with gr.Row():
                    btn_demo = gr.Button("Run demo", variant="secondary")
                    btn = gr.Button("Compute turns", variant="primary")
                status = gr.Textbox(label="Status", lines=7)
                total_df = gr.Dataframe(label="Totals (full recording)")
                blocks_df = gr.Dataframe(label="By 5-minute block")
                with gr.Row():
                    f_total = gr.File(label="conversational_turns_total.csv")
                    f_blocks = gr.File(label="conversational_turns_by_5min.csv")
                    f_detail = gr.File(label="conversational_turns_detail.csv")

                btn.click(
                    _run,
                    inputs=[rttm, upload, out_dir, gap, block],
                    outputs=[status, total_df, blocks_df, f_total, f_blocks, f_detail],
                )
                btn_demo.click(
                    _run_demo,
                    inputs=[out_dir, gap, block],
                    outputs=[status, total_df, blocks_df, f_total, f_blocks, f_detail],
                )

            with gr.Tab("2 · Sapelo2 (VTC labeling)"):
                gr.Markdown(
                    """
VTC speaker labeling needs **Linux + FFmpeg + GPU**. Use UGA **Sapelo2**.

After the job finishes, download `rttm.csv` and use tab **1 · Compute turns**.
"""
                )
                gr.Code(value=_sapelo_text(), language="markdown", label="Sapelo instructions")

            with gr.Tab("3 · About"):
                gr.Markdown(
                    """
### Labels from VTC

- **FEM** — adult female  
- **MAL** — adult male  
- **KCHI** — key child (wearing the recorder)  
- **OCH** — other children (not used for these turn counts)

### Outputs

1. `conversational_turns_total.csv` — totals per recording  
2. `conversational_turns_by_5min.csv` — per 5-min block (zeros included)  
3. `conversational_turns_detail.csv` — one row per turn  

### CLI (no GUI)

```bash
python -m vtc_pipeline.conversational_turns --rttm path/to/rttm.csv --out-dir vtc_runs/my_run
```
"""
                )
    return demo


def main() -> None:
    app = build()
    app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)


if __name__ == "__main__":
    main()
