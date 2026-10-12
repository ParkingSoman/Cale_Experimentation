# Remove Cale (Windows). Finds a Python to run cale\uninstall.py.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($py in @("python", "py")) { if (Get-Command $py -ErrorAction SilentlyContinue) { & $py "$here\uninstall.py" @args; exit $LASTEXITCODE } }
Write-Error "No Python found."; exit 1
