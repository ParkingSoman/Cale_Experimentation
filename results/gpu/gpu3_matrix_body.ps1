$cpu = 'C:\cale\xplat\llama-cpu\llama-bench.exe'
$vk  = 'C:\cale\xplat\llama-vk\llama-bench.exe'
$m   = 'C:\cale\xplat\gguf\gemma-3n-E4B-it-Q4_0.gguf'
$runs = @(
  @('A cpu-build repack1',          $cpu, @('--repack','1')),
  @('B cpu-build repack0',          $cpu, @('--repack','0')),
  @('C vk-build dev-none',          $vk,  @('-dev','none','-ngl','0')),
  @('D vk ngl0 host-on',            $vk,  @('-dev','Vulkan0','-ngl','0','--no-host','0')),
  @('E vk ngl0 NO-HOST',            $vk,  @('-dev','Vulkan0','-ngl','0','--no-host','1')),
  @('F vk ngl16 NO-HOST',           $vk,  @('-dev','Vulkan0','-ngl','16','--no-host','1')),
  @('G vk ngl0 NO-HOST nopo',       $vk,  @('-dev','Vulkan0','-ngl','0','--no-host','1','-nopo','1'))
)
foreach ($r in $runs) {
  "=== $($r[0])"; Get-Date -Format T
  $a = @('-m', $m, '-p', '512', '-n', '64', '-r', '2', '-t', '4') + $r[2]
  & $r[1] @a 2>&1 | ForEach-Object { "$_" } | Select-String -Pattern 'pp512|tg64|rror|fail|out of|repack|REPACK' | Select-Object -First 8
  "exit code: $LASTEXITCODE"
}
"GPU3_DONE"
