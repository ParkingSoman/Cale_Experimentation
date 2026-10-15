# Build Cale-Setup.exe from the files that `python scripts/build_installer.py win` made.
# Run on Windows, in the folder that holds payload.zip, setup.cmd and setup.vbs:
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1
# It uses IExpress, a tool that comes with Windows. If the demo certificate exists (see prep_windows_demo.ps1),
# the .exe is signed with it. No admin rights are needed to build.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $here "Cale-Setup.exe"
$sed = Join-Path $here "Cale.sed"
foreach ($f in "payload.zip", "setup.cmd", "setup.vbs") {
    if (-not (Test-Path (Join-Path $here $f))) { throw "missing $f. Run scripts/build_installer.py win first." }
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
FriendlyName=Cale
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

# IExpress returns at once; wait for the file to appear and stop growing.
$sedArg = if ($sed -match " ") { "`"$sed`"" } else { $sed }          # quotes only when the path has a space: IExpress rejects them otherwise
Start-Process -FilePath "iexpress.exe" -ArgumentList "/N", "/Q", $sedArg -Wait -WindowStyle Hidden
$deadline = (Get-Date).AddMinutes(5)
while (-not (Test-Path $target) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
if (-not (Test-Path $target)) { throw "IExpress did not make $target" }
do { $size = (Get-Item $target).Length; Start-Sleep -Seconds 2 } while ($size -ne (Get-Item $target).Length)

# Sign with the demo certificate, if this computer has one.
$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -eq "CN=Cale Demo" } | Select-Object -First 1
if ($cert) {
    Set-AuthenticodeSignature -FilePath $target -Certificate $cert -HashAlgorithm SHA256 | Out-Null
    Write-Host "signed with the demo certificate"
} else {
    Write-Host "not signed (no demo certificate here). Windows will warn about an unknown publisher."
}
Remove-Item $sed -Force
Write-Host ("built: {0}  ({1:N1} MB)" -f $target, ((Get-Item $target).Length / 1MB))
