# Build the app's setup .exe from the files that `python scripts/build_app.py win APP` made.
# Run on Windows, in the folder that holds payload.zip, name.txt and (for desktop apps) setup.ini, app.ico and host\:
#   powershell -ExecutionPolicy Bypass -File build_app_exe.ps1
# A desktop app (Quick Notes) gets a small self-extracting installer with the app's own icon (compiled with csc.exe, which
# comes with Windows). When it runs, it unpacks the app, adds a Start Menu entry with the icon, and opens the app in its own window.
# Other apps still use IExpress. Signs with the demo certificate if this computer has one (see prep_windows_demo.ps1).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$name = (Get-Content (Join-Path $here "name.txt") -Raw).Trim()
$target = Join-Path $here "$name.exe"
if (-not (Test-Path (Join-Path $here "payload.zip"))) { throw "missing payload.zip. Run scripts/build_app.py win first." }

if ((Test-Path (Join-Path $here "setup.ini")) -and (Test-Path (Join-Path $here "host\SelfExtract.cs"))) {
    . (Join-Path $here "host\pack.ps1")
    $csc = Find-Csc
    if ($csc) {
        $settings = [System.IO.File]::ReadAllText((Join-Path $here "setup.ini"))
        New-SetupExe $csc (Join-Path $here "host") (Join-Path $here "app.ico") ($name -replace "-Setup$", "" -replace "-", " ") $settings (Join-Path $here "payload.zip") $target
        Sign-Demo $target
        Write-Host ("built: {0}  ({1:N1} MB)" -f $target, ((Get-Item $target).Length / 1MB))
        return
    }
    Write-Host "csc.exe not found: using IExpress (default icon, no window)."
}

# ---- IExpress (other apps, or no csc.exe) ----
$sed = Join-Path $here "app.sed"
foreach ($f in "setup.cmd", "setup.vbs") {
    if (-not (Test-Path (Join-Path $here $f))) { throw "missing $f. Run scripts/build_app.py win first." }
}
if (Test-Path $target) { Remove-Item $target -Force }
@"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=%InstallPrompt%
DisplayLicense=%DisplayLicense%
FinishMessage=%FinishMessage%
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=%PostInstallCmd%
AdminQuietInstCmd=%AdminQuietInstCmd%
UserQuietInstCmd=%UserQuietInstCmd%
SourceFiles=SourceFiles
[Strings]
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=$target
FriendlyName=$name
AppLaunched=wscript.exe setup.vbs
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
FILE0="payload.zip"
FILE1="setup.cmd"
FILE2="setup.vbs"
[SourceFiles]
SourceFiles0=$here\
[SourceFiles0]
%FILE0%=
%FILE1%=
%FILE2%=
"@ | Set-Content -Path $sed -Encoding ASCII
$sedArg = if ($sed -match " ") { "`"$sed`"" } else { $sed }
Start-Process -FilePath "iexpress.exe" -ArgumentList "/N", "/Q", $sedArg -Wait -WindowStyle Hidden
$deadline = (Get-Date).AddMinutes(5)
while (-not (Test-Path $target) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
if (-not (Test-Path $target)) { throw "IExpress did not make $target" }
do { $size = (Get-Item $target).Length; Start-Sleep -Seconds 2 } while ($size -ne (Get-Item $target).Length)
$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -eq "CN=Cale Demo" } | Select-Object -First 1
if ($cert) { Set-AuthenticodeSignature -FilePath $target -Certificate $cert -HashAlgorithm SHA256 | Out-Null; Write-Host "signed with the demo certificate" }
else { Write-Host "not signed (no demo certificate here)" }
Remove-Item $sed -Force
Write-Host ("built: {0}  ({1:N1} MB)" -f $target, ((Get-Item $target).Length / 1MB))
