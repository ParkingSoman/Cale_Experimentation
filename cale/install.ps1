# Install Cale on Windows. Run once:  powershell -ExecutionPolicy Bypass -File cale\install.ps1
# This only finds a Python to run cale\install.py. If the computer has none, it gets one from uv.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($py in @("python", "py")) {
  $cmd = Get-Command $py -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
    & $py -c "import sys; sys.exit(sys.version_info < (3, 9))" 2>$null
    if ($LASTEXITCODE -eq 0) { & $py "$here\install.py" @args; exit $LASTEXITCODE }
  }
}
Write-Host "==> No Python 3.9 or newer found. Getting one with uv."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression }
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv run --python 3.12 python "$here\install.py" @args
exit $LASTEXITCODE
