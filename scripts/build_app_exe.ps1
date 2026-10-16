# Build the app's setup .exe from the files that `python scripts/build_app.py win APP` made.
# Run on Windows, in the folder that holds payload.zip, setup.cmd, setup.vbs and name.txt:
#   powershell -ExecutionPolicy Bypass -File build_app_exe.ps1
# Uses IExpress (part of Windows). Signs with the demo certificate if this computer has one (see prep_windows_demo.ps1).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$name = (Get-Content (Join-Path $here "name.txt") -Raw).Trim()
$target = Join-Path $here "$name.exe"
$sed = Join-Path $here "app.sed"
foreach ($f in "payload.zip", "setup.cmd", "setup.vbs") {
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
