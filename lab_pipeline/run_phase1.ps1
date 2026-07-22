# Lab pipeline — Phase 1 (review CSVs + clips)
# Edit CONFIG if needed, then double-click or run in PowerShell.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # repo root if script lives in lab_pipeline/
# If this file is in lab_pipeline/, parent is repo root:
Set-Location $PSScriptRoot\..

$CONFIG = "lab_pipeline\config.yaml"

Write-Host "Running Phase 1 with $CONFIG"
python -m lab_pipeline.run_all --config $CONFIG
Write-Host "Done. See run_dir/output in your config."
Pause
