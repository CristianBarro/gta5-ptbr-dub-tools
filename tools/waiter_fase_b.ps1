# Starts the cutscene phase B as soon as lote_v1 has finished GENERATING Trevor (GPU free), instead of waiting for
# the rest of run_lote (its ASR / speaker / fit steps run on the CPU and keep going in parallel).
# Before phase B, a short automatic test of batched Trevor generation (Qwen) runs on the free GPU:
# 24 lines one by one vs batches of 8 and 16; transcribed and judged on the CPU while phase B generates
# Michael/Franklin. phase B uses the batch size only if the verdict passed (see qwen_batch_eval.py).
$ErrorActionPreference = 'Continue'
$G     = 'C:\gta-dub'
$Tools = "$G\work\gta-dialogue"
$Lote  = "$G\outputs\GTA_V_Dublagem_PTBR\cutscenes_lote"
$Drv   = "$G\outputs\GTA_V_Dublagem_PTBR\lote_v1\logs\driver.log"
$Appr  = "$G\outputs\GTA_V_Dublagem_PTBR\revisao_v5\approval.json"
$Test  = "$Lote\qwen_batch_test"
$PyGen = 'D:\JARVIS\.venv-qwen\Scripts\python.exe'
$PyAsr = 'D:\JARVIS\.venv\Scripts\python.exe'
$Py    = 'C:\Program Files\Python310\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
function Note($t) { "$(Get-Date -Format s) $t" | Add-Content "$Lote\STATUS.txt" }
function TrevorGenRunning { [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*dub_batch.py*--character TREVOR*' -and $_.CommandLine -like '*lote_v1*' }) }
Note "waiter (pid $PID): espera o lote_v1 terminar de GERAR o Trevor (otimizacao 1)"
while ($true) {
  $done = (Test-Path $Drv) -and (Select-String -Path $Drv -Pattern 'GEN TREVOR exit 0' -Quiet)
  $loteAlive = [bool](Get-Process -Id 29912 -ErrorAction SilentlyContinue)
  if ($done -and -not (TrevorGenRunning)) { break }
  if (-not $loteAlive -and -not (TrevorGenRunning)) { break }     # run_lote ended (or died): GPU free anyway
  Start-Sleep -Seconds 30
}
Note 'Trevor das missoes gerado - GPU livre; teste de lote do Qwen'
New-Item -ItemType Directory -Force $Test | Out-Null
& $Py "$Tools\qwen_batch_eval.py" queue $Lote $Test 1> "$Test\queue.log" 2>&1
$sizes = @(8, 16)
if (Test-Path "$Test\fila_teste.csv") {
  & $PyGen "$Tools\dub_batch.py" --queue "$Test\fila_teste.csv" --approval $Appr --out "$Test\single" --character TREVOR --select all --takes 1 --batch-size 1 1> "$Test\single.log" 2> "$Test\single.err"
  foreach ($bs in $sizes) {
    & $PyGen "$Tools\dub_batch.py" --queue "$Test\fila_teste.csv" --approval $Appr --out "$Test\batch$bs" --character TREVOR --select all --takes 1 --batch-size $bs 1> "$Test\batch$bs.log" 2> "$Test\batch$bs.err"
  }
  Note 'teste de lote do Qwen: geracao feita; avaliacao na CPU em paralelo com a fase B'
} else { Note 'teste de lote do Qwen: fila de teste nao criada - Trevor fica 1 por vez' }
Note 'iniciando fase B'
$b = Start-Process -FilePath 'powershell' -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', `
  "& '$Tools\run_cutscenes.ps1' -Phase B -NoLote -SepThreads 8 *> '$Lote\logs\phaseB_runner.log'" -WindowStyle Hidden -PassThru
Note "fase B pid $($b.Id)"
if (Test-Path "$Test\fila_teste.csv") {
  foreach ($d in @('single') + ($sizes | ForEach-Object { "batch$_" })) {
    if (Test-Path "$Test\$d\manifest.json") { & $PyAsr -I "$Tools\cut_pick.py" "$Test\$d" --model medium 1> "$Test\pick_$d.log" 2> "$Test\pick_$d.err" }
  }
  & $Py "$Tools\qwen_batch_eval.py" eval $Test @sizes 1> "$Test\eval.log" 2>&1
  $v = if (Test-Path "$Test\qwen_batch_verdict.json") { Get-Content "$Test\qwen_batch_verdict.json" -Raw | ConvertFrom-Json } else { $null }
  if ($v -and $v.pass) { Note "teste de lote do Qwen PASSOU: Trevor das cutscenes em lotes de $($v.batch_size)" }
  else { Note 'teste de lote do Qwen nao passou: Trevor das cutscenes 1 por vez (como antes)' }
}
