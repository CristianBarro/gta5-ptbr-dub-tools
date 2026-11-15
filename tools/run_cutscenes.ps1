# Cutscene dubbing batch (Michael, Franklin, Trevor in one remix per cutscene).
#   -Phase A   CPU only (safe while the GPU is in use by a game):
#              dialogue-channel separation -> speaker attribution -> generation queue
#   -Phase B   needs the GPU:
#              generation (1 take) on the GPU, while on the CPU the remaining channels of the AWCs that have dubbed
#              lines are separated and the takes are transcribed/picked as they appear;
#              then retake of the bad lines -> pick -> remix per AWC -> patch RPF copies;
#              then, unless -NoLote, resume the in-game line batch lote_v1 (run_lote.ps1: Franklin, Trevor).
# Nothing is installed in the game: the patched RPFs go to <Lote>\pacote\mods\... for a later test install.
#   -MainDevice cpu|cuda : device for the phase-A separation (cpu = idle priority, 4 threads, GPU left alone)
param([ValidateSet('A', 'B')] [string]$Phase = 'B', [switch]$NoLote, [int]$SepThreads = 10, [string]$SepDevice = 'cpu',
      [ValidateSet('cpu', 'cuda')] [string]$MainDevice = 'cpu')
# 'Continue': under an outer *> redirect, Windows PowerShell 5.1 turns any stderr line of a native command (e.g. a
# FutureWarning from diffusers) into a terminating error with 'Stop'. Failures are caught by Step via $LASTEXITCODE.
$ErrorActionPreference = 'Continue'
$Tools = 'C:\gta-dub\work\gta-dialogue'
$Lote  = 'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR\cutscenes_lote'
$Work  = 'D:\gta_dub_work\cut'
$Appr  = 'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR\revisao_v5\approval.json'
$Game  = 'D:\SteamLibrary\steamapps\common\Grand Theft Auto V'
$PyGen = 'D:\JARVIS\.venv-qwen\Scripts\python.exe'
$PyAsr = 'D:\JARVIS\.venv\Scripts\python.exe'
$PyTts = 'D:\JARVIS\.venv-tts\Scripts\python.exe'
$Py    = 'C:\Program Files\Python310\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$Logs = Join-Path $Lote 'logs'; New-Item -ItemType Directory -Force $Logs | Out-Null
function Note($text) { "$(Get-Date -Format s) $text" | Add-Content "$Lote\STATUS.txt" }
function Step($name, [scriptblock]$cmd) {
  Note "START $name"
  & $cmd
  if ($LASTEXITCODE -ne 0) { Note "FAILED $name (exit $LASTEXITCODE)"; throw "$name failed" }
  Note "DONE  $name"
}
function SepRunning { [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*cut_separate.py*' }) }

if ($Phase -eq 'A') {
  if ($MainDevice -eq 'cpu') { (Get-Process -Id $PID).PriorityClass = 'Idle' }   # children inherit: the game always comes first
  Note "PHASE A ($MainDevice): waiting for the dialogue-channel separation"
  while (SepRunning) { Start-Sleep -Seconds 30 }
  # complete whatever a stopped run left (resumable), still on the CPU
  if ($MainDevice -eq 'cpu') { $env:CUT_SEP_DEVICE = 'cpu'; $env:CUT_SEP_THREADS = '4' } else { $env:CUT_SEP_DEVICE = ''; $env:CUT_SEP_THREADS = '' }
  Step 'separate main' { & $PyGen "$Tools\cut_separate.py" --batch-main "$Work\orig" "$Work\voice" 1>> "$Work\separate_main.log" 2>> "$Work\separate_main.err" }
  Step 'speakers' { & $PyTts "$Tools\cut_speakers_batch.py" $Lote "$Work\voice" 1> "$Logs\speakers.log" 2> "$Logs\speakers.err" }
  Step 'queue' { & $Py "$Tools\cut_queue_batch.py" $Lote 1> "$Logs\queue.log" 2> "$Logs\queue.err" }
  Note 'PHASE A READY - falantes e fila prontos; fase B precisa da GPU'
  return
}

# ---- Phase B ----
# keep the PC awake while phase B runs (as run_lote.ps1 does); no settings changed
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([uint32]'0x80000001')
if (-not (Test-Path "$Lote\fila_gen.csv")) { throw 'Run -Phase A first (fila_gen.csv missing).' }
Remove-Item "$Lote\gen_done.flag" -ErrorAction SilentlyContinue
# CPU, in parallel with the GPU generation: remaining channels of the AWCs that will be remixed
$env:CUT_SEP_DEVICE = $SepDevice; $env:CUT_SEP_THREADS = "$SepThreads"
# a separate-rest run started earlier (on the CPU, before the GPU was free) is reused instead of starting a second one
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*cut_separate.py*--batch-rest*' } | Select-Object -First 1
if ($running) {
  $sep = Get-Process -Id $running.ProcessId
  Note "separate rest already running (pid $($sep.Id)) - reused"
} else {
  $sep = Start-Process -FilePath $PyGen -ArgumentList "$Tools\cut_separate.py", '--batch-rest', "$Work\orig", "$Work\voice", "$Lote\awc_com_falas.txt" `
    -RedirectStandardOutput "$Work\separate_rest.log" -RedirectStandardError "$Work\separate_rest.err" -WindowStyle Hidden -PassThru
  Note "START separate rest (pid $($sep.Id), $SepDevice, $SepThreads threads)"
}
# Trevor (Qwen) in batches only if the automatic batch test passed (qwen_batch_verdict.json, written by qwen_batch_eval.py)
function TrevorBatch {
  $v = "$Lote\qwen_batch_test\qwen_batch_verdict.json"
  if (Test-Path $v) { $j = Get-Content $v -Raw | ConvertFrom-Json; if ($j.pass) { return [int]$j.batch_size } }
  return 1
}
# CPU, in parallel: transcribe and pick takes as they are generated (incremental; stops after the flag appears)
$pickLoop = Start-Process -FilePath 'powershell' -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', `
  "`$env:PYTHONIOENCODING='utf-8'; while (-not (Test-Path '$Lote\gen_done.flag')) { if (Test-Path '$Lote\gen\manifest.json') { & '$PyAsr' -I '$Tools\cut_pick.py' '$Lote\gen' --model medium 1>> '$Logs\pick_loop.log' 2>> '$Logs\pick_loop.err' }; Start-Sleep -Seconds 120 }" `
  -WindowStyle Hidden -PassThru
foreach ($who in 'MICHAEL', 'FRANKLIN', 'TREVOR') {
  $bs = if ($who -eq 'TREVOR') { TrevorBatch } else { 1 }
  Step "gen $who (batch $bs)" { & $PyGen "$Tools\dub_batch.py" --queue "$Lote\fila_gen.csv" --approval $Appr --out "$Lote\gen" --character $who --select all --takes 1 --batch-size $bs 1> "$Logs\gen_$who.log" 2> "$Logs\gen_$who.err" }
}
'done' | Set-Content "$Lote\gen_done.flag"
$pickLoop.WaitForExit()
Step 'pick 1' { & $PyAsr -I "$Tools\cut_pick.py" "$Lote\gen" --model medium 1> "$Logs\pick1.log" 2> "$Logs\pick1.err" }
Step 'retake queue' { & $Py "$Tools\cut_queue_batch.py" $Lote --retake "$Lote\gen\escolha.json" 1> "$Logs\retake_queue.log" 2> "$Logs\retake_queue.err" }
foreach ($who in 'MICHAEL', 'FRANKLIN', 'TREVOR') {
  $bs = if ($who -eq 'TREVOR') { TrevorBatch } else { 1 }
  Step "retake $who (batch $bs)" { & $PyGen "$Tools\dub_batch.py" --queue "$Lote\fila_retake.csv" --approval $Appr --out "$Lote\gen" --character $who --select all --takes 2 --batch-size $bs 1> "$Logs\retake_$who.log" 2> "$Logs\retake_$who.err" }
}
Step 'pick 2' { & $PyAsr -I "$Tools\cut_pick.py" "$Lote\gen" --model medium 1> "$Logs\pick2.log" 2> "$Logs\pick2.err" }
Note 'waiting for separate rest'
$sep.WaitForExit()
while (SepRunning) { Start-Sleep -Seconds 30 }   # a second CPU worker (reversed list) may still be running
# the GPU is free now: finish any channel the CPU run did not reach (resumable, normally nothing left)
$env:CUT_SEP_DEVICE = ''; $env:CUT_SEP_THREADS = ''
Step 'separate rest (finish)' { & $PyGen "$Tools\cut_separate.py" --batch-rest "$Work\orig" "$Work\voice" "$Lote\awc_com_falas.txt" 1>> "$Work\separate_rest.log" 2>> "$Work\separate_rest.err" }
Step 'mix' { & $PyTts "$Tools\cut_mix_batch.py" $Lote $Work 1> "$Logs\mix.log" 2> "$Logs\mix.err" }
Step 'patch' { & "$Tools\awcpatcher\bin\AwcPatcher.exe" patch $Game "$Lote\pacote\mods" "$Lote\plan_cutscenes.tsv" 1> "$Logs\patch.log" 2> "$Logs\patch.err" }
Note "CUTSCENES READY - pacote em $Lote\pacote (nao instalado)"
if (-not $NoLote) { & powershell -ExecutionPolicy Bypass -File "$Tools\run_lote.ps1" }
