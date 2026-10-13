$vk = 'C:\cale\xplat\llama-vk\llama-bench.exe'
$m  = 'C:\cale\xplat\gguf\gemma-3n-E4B-it-Q4_0.gguf'
$runs = @(
  @('V1 cpu-only',                 @('-dev','none','-ngl','0')),
  @('V2 mx150 fit300 NO-HOST',     @('-dev','Vulkan1','-ngl','99','-fitt','300','--no-host','1')),
  @('V3 cpu-only again',           @('-dev','none','-ngl','0')),
  @('V4 mx150 fit300 NO-HOST again', @('-dev','Vulkan1','-ngl','99','-fitt','300','--no-host','1'))
)
foreach ($r in $runs) {
  "=== $($r[0])"; Get-Date -Format T
  $a = @('-m', $m, '-p', '512', '-n', '128', '-r', '3', '-t', '4') + $r[1]
  & $vk @a 2>&1 | ForEach-Object { "$_" } | Select-String -Pattern 'pp512|tg128|rror|fail|out of' | Select-Object -First 6
  "exit code: $LASTEXITCODE"
}
"GPU7_DONE"
