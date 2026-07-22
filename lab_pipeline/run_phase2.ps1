# Lab pipeline — Phase 2 (episodes from filled review notes)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$CONFIG = "lab_pipeline\config.yaml"
# Edit this path after RAs fill the notes CSV:
$NOTES = "lab_runs\full_pipeline_run\output\review_notes_consecutive.csv"

if (-not (Test-Path $NOTES)) {
    Write-Host "Review notes not found: $NOTES"
    Write-Host "Fill is_true_distress in the notes CSV first, then update `$NOTES in this script."
    Pause
    exit 1
}

Write-Host "Running Phase 2 with notes: $NOTES"
python -m lab_pipeline.run_all --config $CONFIG --skip-phase1 --phase2 --review-notes $NOTES
Write-Host "Done. See output/phase2/"
Pause
