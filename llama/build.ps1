# Build cale_llama.dll for Windows: fetch the pinned llama.cpp, apply our patch, build with a portable GCC (w64devkit).
# No admin rights and no Visual Studio needed. usage: powershell -ExecutionPolicy Bypass -File llama\build.ps1 [-Out C:\cale\lbuild] [-Python python.exe]
param([string]$Out = "$PSScriptRoot\build", [string]$Python = "python", [string]$BuildName = "b", [string]$CmakeExtra = "-DGGML_OPENMP=OFF")
# OpenMP is off on purpose: the mingw OpenMP runtime made answers 43-50% slower on the same CPU (measured 2026-10-12:
# smallest size 6.2 -> 12.3 tokens/s, full size 4.1 -> 7.2). llama.cpp's own thread pool is as fast as OpenMP on Linux.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Tag = "b11558"; $Kit = "v2.10.0"
New-Item -ItemType Directory -Force $Out | Out-Null
$kitDir = Join-Path $Out "w64devkit"
if (-not (Test-Path "$kitDir\bin\gcc.exe")) {
  $exe = Join-Path $Out "w64devkit.7z.exe"
  curl.exe -L -s --retry 5 --limit-rate 20m -o $exe "https://github.com/skeeto/w64devkit/releases/download/$Kit/w64devkit-x64-2.10.0.7z.exe"
  & $exe "-o$Out" -y | Out-Null                       # self-extracting archive: makes $Out\w64devkit
}
$src = Join-Path $Out "llama.cpp-$Tag"                # the folder the zip makes. No rename: the virus scanner may hold it.
if (-not (Test-Path "$src\CMakeLists.txt")) {
  $zip = Join-Path $Out "llama.zip"
  curl.exe -L -s --retry 5 --limit-rate 20m -o $zip "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$Tag.zip"
  Expand-Archive $zip -DestinationPath $Out -Force
}
if (-not (Test-Path "$src\.cale-patched")) {
  Get-ChildItem "$PSScriptRoot\patches\*.patch" | ForEach-Object {
    Push-Location $src; & "$kitDir\bin\patch.exe" -p1 -i $_.FullName; Pop-Location
    if ($LASTEXITCODE -ne 0) { throw "patch failed: $($_.Name)" }
  }
  Set-Content "$src\.cale-patched" "applied"
}
& $Python -m pip install --quiet --disable-pip-version-check cmake ninja
$scripts = Join-Path (Split-Path $Python) "Scripts"
$env:PATH = "$kitDir\bin;$scripts;$env:PATH"
$b = Join-Path $Out $BuildName
cmake -S $PSScriptRoot -B $b -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ "-DLLAMA_CPP_DIR=$src" $CmakeExtra.Split(" ", [StringSplitOptions]::RemoveEmptyEntries) | Out-Null
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build $b --target cale_llama -j 8
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Get-ChildItem $b -Filter cale_llama.dll -Recurse | Select-Object -ExpandProperty FullName
