# Launch the lab GUI from the repo root
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Installing basic packages (if needed)..."
python -m pip install -q -r requirements.txt

Write-Host "Starting GUI at http://127.0.0.1:7860 ..."
python -m lab_pipeline.gui_app
