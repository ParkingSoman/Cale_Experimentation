$vk = 'C:\cale\xplat\llama-vk\llama-bench.exe'
$m  = 'C:\cale\xplat\gguf\gemma-3n-E4B-it-Q4_0.gguf'
$runs = @(
  @('H ngl99 fit300 NO-HOST',        @('-ngl','99','-fitt','300','--no-host','1')),
  @('I ngl24 ub256 NO-HOST',         @('-ngl','24','-ub','256','--no-host','1')),
  @('J ngl28 ub256 NO-HOST',         @('-ngl','28','-ub','256','--no-host','1')),
  @('K ngl35 ub128 NO-HOST',         @('-ngl','35','-ub','128','--no-host','1')),
  @('L ngl24 ub256 host-on',         @('-ngl','24','-ub','256','--no-host','0'))
)
foreach ($r in $runs) {
  "=== $($r[0])"; Get-Date -Format T
  $a = @('-m', $m, '-dev', 'Vulkan0', '-p', '512', '-n', '64', '-r', '2', '-t', '4') + $r[1]
  & $vk @a 2>&1 | ForEach-Object { "$_" } | Select-String -Pattern 'pp512|tg64|rror|fail|out of|fit|layers' | Select-Object -First 10
  "exit code: $LASTEXITCODE"
}
"GPU4_DONE"
