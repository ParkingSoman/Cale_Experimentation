# Helpers for building Cale's Windows programs with the C# compiler that comes with Windows (csc.exe, no install needed).
# Used by build_exe.ps1 (Cale-Setup.exe) and build_app_exe.ps1 (an app's setup exe). Dot-source it:  . .\host\pack.ps1
# Windows PowerShell 5.1 is enough.

function Find-Csc {
    foreach ($d in "Framework64", "Framework") {
        $p = Join-Path $env:WINDIR "Microsoft.NET\$d\v4.0.30319\csc.exe"
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Invoke-Csc($csc, [string[]]$arguments) {
    $out = & $csc /nologo /utf8output /optimize+ @arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw ("csc failed:`n" + ($out -join "`n")) }
}

# Compile CaleWindow.exe and CaleSetup.exe into $binDir. $hostDir holds the .cs files and lib\ (the WebView2 DLLs).
function Build-Host($csc, $hostDir, $icon, $binDir) {
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    $lib = Join-Path $hostDir "lib"
    $core = Join-Path $lib "Microsoft.Web.WebView2.Core.dll"
    $forms = Join-Path $lib "Microsoft.Web.WebView2.WinForms.dll"
    $args1 = @("/target:winexe", "/platform:x64", "/win32icon:$icon", "/out:$(Join-Path $binDir 'CaleWindow.exe')",
        "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", "/r:$core", "/r:$forms",
        "/resource:$core,Microsoft.Web.WebView2.Core.dll", "/resource:$forms,Microsoft.Web.WebView2.WinForms.dll",
        (Join-Path $hostDir "CaleWindow.cs"))
    Invoke-Csc $csc $args1
    $args2 = @("/target:winexe", "/platform:anycpu", "/win32icon:$icon", "/out:$(Join-Path $binDir 'CaleSetup.exe')",
        "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", "/r:Microsoft.CSharp.dll",
        (Join-Path $hostDir "CaleSetup.cs"))
    Invoke-Csc $csc $args2
    Copy-Item (Join-Path $lib "WebView2Loader.dll") (Join-Path $binDir "WebView2Loader.dll") -Force
}

# Put files into a zip made by Python, replacing entries of the same name. $files: @{ "bin/CaleWindow.exe" = "C:\path\file" }
function Add-ToZip($zipPath, $files) {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Update)
    try {
        foreach ($name in $files.Keys) {
            $old = $zip.GetEntry($name)
            if ($old) { $old.Delete() }
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $files[$name], $name, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally { $zip.Dispose() }
}

# Make the self-extracting setup exe: compile SelfExtract.cs with the icon, then add the zip and the settings to its end.
# $title is the name Windows shows for the file. $settings is the text described at the top of SelfExtract.cs.
function New-SetupExe($csc, $hostDir, $icon, $title, $settings, $zipPath, $outPath) {
    $info = Join-Path ([System.IO.Path]::GetTempPath()) ("info-" + [guid]::NewGuid().ToString("N") + ".cs")
    $safe = $title.Replace('"', "'")
    $text = "using System.Reflection;`n[assembly: AssemblyTitle(""$safe"")]`n[assembly: AssemblyProduct(""$safe"")]`n[assembly: AssemblyDescription(""$safe"")]`n[assembly: AssemblyFileVersion(""1.0.0.0"")]`n[assembly: AssemblyVersion(""1.0.0.0"")]`n"
    [System.IO.File]::WriteAllText($info, $text, (New-Object System.Text.UTF8Encoding $false))
    if (Test-Path $outPath) { Remove-Item $outPath -Force }
    try {
        Invoke-Csc $csc @("/target:winexe", "/platform:anycpu", "/win32icon:$icon", "/out:$outPath",
            "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", "/r:System.IO.Compression.dll", "/r:System.IO.Compression.FileSystem.dll",
            (Join-Path $hostDir "SelfExtract.cs"), $info)
    } finally { Remove-Item $info -Force -ErrorAction SilentlyContinue }
    $cfg = [System.Text.Encoding]::UTF8.GetBytes($settings)
    $zipLen = (Get-Item $zipPath).Length
    $out = [System.IO.File]::Open($outPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
    try {
        $in = [System.IO.File]::OpenRead($zipPath)
        try { $in.CopyTo($out) } finally { $in.Dispose() }
        $out.Write($cfg, 0, $cfg.Length)
        $w = New-Object System.IO.BinaryWriter($out)
        $w.Write([int32]$cfg.Length)
        $w.Write([int64]$zipLen)
        $w.Write([System.Text.Encoding]::ASCII.GetBytes("CALEZIP1"))
        $w.Flush()
    } finally { $out.Dispose() }
}

# Sign with the demo certificate, if this computer has one (see prep_windows_demo.ps1). Sign AFTER the zip is added.
function Sign-Demo($path) {
    $cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue | Where-Object { $_.Subject -eq "CN=Cale Demo" } | Select-Object -First 1
    if ($cert) {
        Set-AuthenticodeSignature -FilePath $path -Certificate $cert -HashAlgorithm SHA256 | Out-Null
        Write-Host "signed with the demo certificate"
    } else {
        Write-Host "not signed (no demo certificate here). Windows will warn about an unknown publisher."
    }
}
