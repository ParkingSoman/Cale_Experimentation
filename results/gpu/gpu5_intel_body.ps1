$vk = 'C:\cale\xplat\llama-vk\llama-bench.exe'
$m  = 'C:\cale\xplat\gguf\gemma-3n-E4B-it-Q4_0.gguf'
"--- devices"; & $vk --list-devices 2>&1 | ForEach-Object { "$_" } | Select-Object -First 12
$runs = @(
  @('M intel ngl0 host-on',      @('-dev','Vulkan1','-ngl','0','--no-host','0')),
  @('N intel ngl0 NO-HOST',      @('-dev','Vulkan1','-ngl','0','--no-host','1')),
  @('O intel ngl99 host-on',     @('-dev','Vulkan1','-ngl','99','--no-host','0')),
  @('P intel ngl99 NO-HOST',     @('-dev','Vulkan1','-ngl','99','--no-host','1')),
  @('Q nvidia fit300 NO-HOST',   @('-dev','Vulkan0','-ngl','99','-fitt','300','--no-host','1'))
)
foreach ($r in $runs) {
  "=== $($r[0])"; Get-Date -Format T
  $a = @('-m', $m, '-p', '512', '-n', '64', '-r', '2', '-t', '4') + $r[1]
  & $vk @a 2>&1 | ForEach-Object { "$_" } | Select-String -Pattern 'pp512|tg64|rror|fail|out of|xception|violation' | Select-Object -First 8
  "exit code: $LASTEXITCODE"
}
"GPU5_DONE"
