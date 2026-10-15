# ONE-TIME prep of a Windows DEMO computer, so that a downloaded Cale-Setup.exe opens with no warning.
# This lowers this computer's protection. Use it only on a computer you use for demos. Undo it with: -Undo
#   Run as administrator:   powershell -ExecutionPolicy Bypass -File prep_windows_demo.ps1
#   Undo:                   powershell -ExecutionPolicy Bypass -File prep_windows_demo.ps1 -Undo
# What it changes:
#   1. Makes a code-signing certificate "CN=Cale Demo" (self-signed) and trusts it as a root and as a publisher.
#   2. Turns off SmartScreen "Check apps and files" for this computer.
# A public release needs a real certificate (for example Azure Trusted Signing). This is for the demo only.
param([switch]$Undo)
$ErrorActionPreference = "Stop"
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this as administrator." }
$key = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer"
$backup = "HKLM:\SOFTWARE\CaleDemoPrep"

if ($Undo) {
    foreach ($store in "Cert:\LocalMachine\Root", "Cert:\LocalMachine\TrustedPublisher", "Cert:\CurrentUser\My") {
        Get-ChildItem $store | Where-Object { $_.Subject -eq "CN=Cale Demo" } | Remove-Item -Force
    }
    if (Test-Path $backup) {
        $old = (Get-ItemProperty $backup).SmartScreenEnabled
        Set-ItemProperty -Path $key -Name SmartScreenEnabled -Value $old
        Remove-Item $backup -Recurse -Force
    }
    Write-Host "Undone: the demo certificate is removed and SmartScreen is back as it was."
    exit 0
}

if (-not (Test-Path $backup)) {                                   # remember what it was, for -Undo
    New-Item $backup -Force | Out-Null
    $cur = (Get-ItemProperty $key -ErrorAction SilentlyContinue).SmartScreenEnabled
    if (-not $cur) { $cur = "Warn" }
    Set-ItemProperty -Path $backup -Name SmartScreenEnabled -Value $cur
}
$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -eq "CN=Cale Demo" } | Select-Object -First 1
if (-not $cert) {
    $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=Cale Demo" -CertStoreLocation Cert:\CurrentUser\My -NotAfter (Get-Date).AddYears(2)
}
$cer = Join-Path $env:TEMP "cale-demo.cer"
Export-Certificate -Cert $cert -FilePath $cer | Out-Null
Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
Remove-Item $cer -Force
Set-ItemProperty -Path $key -Name SmartScreenEnabled -Value "Off"
Write-Host "Done. Build Cale-Setup.exe (build_exe.ps1) as the same user: it will be signed with the demo certificate."
