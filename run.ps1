# =============================================================================
# CVision — Launcher Script
# Run this from the Resume Ranker directory to start the app.
#
# Usage (from PowerShell in the project folder):
#     .\run.ps1
# =============================================================================

$venvPython  = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
$appPath     = Join-Path $PSScriptRoot "app.py"

# Verify venv exists
if (-not (Test-Path $venvPython)) {
    Write-Host "❌ Virtual environment not found." -ForegroundColor Red
    Write-Host "   Create it first:" -ForegroundColor Yellow
    Write-Host "   C:\Users\talha\Anaconda3\python.exe -m venv .venv" -ForegroundColor Cyan
    Write-Host "   .\.venv\Scripts\pip.exe install -r requirements.txt" -ForegroundColor Cyan
    exit 1
}

# Show which Python is being used
$pyVersion = & $venvPython --version 2>&1
Write-Host "✅ Using: $pyVersion (Python 3.12 venv)" -ForegroundColor Green

# Launch Streamlit
Write-Host "🚀 Starting CVision..." -ForegroundColor Cyan
& $venvStreamlit run $appPath
