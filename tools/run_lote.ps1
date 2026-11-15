# Overnight PT-BR dubbing batch: generation (GPU) + QA (CPU, in parallel) + duration fit.
# Runs detached from Claude; resumable: run it again and it continues where it stopped.
# Keeps the PC awake only while this script runs (SetThreadExecutionState); no settings are changed.
param(
  [string]$Out = 'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR\lote_v1',
  [int]$Takes = 1,
  [string]$Select = 'all',
  [int]$PerCharacter = 12
)
$ErrorActionPreference = 'Continue'
$G     = 'C:\gta-dub'
$Tools = "$G\work\gta-dialogue"
$Queue = "$G\outputs\GTA_V_Dialogue_Map_v0.1\fila_dublagem.csv"
$Appr  = "$G\outputs\GTA_V_Dublagem_PTBR\revisao_v5\approval.json"
$PyGen = 'D:\JARVIS\.venv-qwen\Scripts\python.exe'
$PyAsr = 'D:\JARVIS\.venv\Scripts\python.exe'
$PyTts = 'D:\JARVIS\.venv-tts\Scripts\python.exe'
$Logs  = "$Out\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$env:PYTHONIOENCODING = 'utf-8'

Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([uint32]'0x80000001')   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function Log($m) { $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"; Add-Content -LiteralPath "$Logs\driver.log" -Value $line -Encoding UTF8 }
function Start-Asr($tag) {
  Start-Process -FilePath $PyAsr -ArgumentList @("$Tools\dub_qa.py", 'asr', "`"$Out`"") -NoNewWindow -PassThru `
    -RedirectStandardOutput "$Logs\asr_$tag.log" -RedirectStandardError "$Logs\asr_$tag.err"
}
"running $(Get-Date -Format s) pid $PID" | Set-Content -LiteralPath "$Out\STATUS.txt" -Encoding UTF8
Log "START out=$Out takes=$Takes select=$Select"
$asr = $null
foreach ($who in 'MICHAEL', 'FRANKLIN', 'TREVOR') {
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    Log "GEN $who attempt $attempt"
    & $PyGen "$Tools\dub_batch.py" --queue $Queue --approval $Appr --out $Out --character $who --select $Select `
        --per-character $PerCharacter --takes $Takes 1>> "$Logs\gen_$who.log" 2>> "$Logs\gen_$who.err"
    Log "GEN $who exit $LASTEXITCODE"
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 20
  }
  if ($asr -and -not $asr.HasExited) { $asr.WaitForExit() }
  $asr = Start-Asr $who      # transcribes what exists so far while the next character generates
}
if ($asr -and -not $asr.HasExited) { $asr.WaitForExit() }
Log 'QA asr final'
& $PyAsr "$Tools\dub_qa.py" asr $Out 1>> "$Logs\asr_final.log" 2>> "$Logs\asr_final.err"
Log 'QA speaker'
& $PyTts "$Tools\dub_qa.py" speaker $Out 1>> "$Logs\speaker.log" 2>> "$Logs\speaker.err"
Log 'QA report'
& 'C:\Program Files\Python310\python.exe' "$Tools\dub_qa.py" report $Out 1> "$Logs\report.log" 2> "$Logs\report.err"
Log 'FIT'
& $PyTts "$Tools\dub_fit.py" $Out 1> "$Logs\fit.log" 2> "$Logs\fit.err"
Log 'DONE'
"done $(Get-Date -Format s)" | Set-Content -LiteralPath "$Out\STATUS.txt" -Encoding UTF8
[void][Win.Power]::SetThreadExecutionState([uint32]'0x80000000')
