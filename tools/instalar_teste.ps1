# Install / remove the PT-BR dubbing TEST package in GTA V Legacy, reversibly.
#   instalar_teste.ps1 -Pacote <pacote dir>            install
#   instalar_teste.ps1 -Pacote <pacote dir> -Remover   remove exactly what was installed
# What it adds (never overwrites an existing file; original RPFs are never touched):
#   <game>\mods\x64\audio\sfx\SS_*.rpf      patched copies from the package
# Prerequisite (done by the user in OpenIV, not by this script): Tools > ASI Manager > install "ASI Loader" and "OpenIV.ASI".
# The script only checks that <game>\dinput8.dll and <game>\OpenIV.asi exist and are real PE files.
# A log with SHA-256 of every added file is kept in <pacote>\instalacao.json; removal deletes only files whose hash matches.
param(
  [Parameter(Mandatory)] [string]$Pacote,
  [string]$Game = 'D:\SteamLibrary\steamapps\common\Grand Theft Auto V',
  [switch]$Remover
)
$ErrorActionPreference = 'Stop'
$log = Join-Path $Pacote 'instalacao.json'
if (Get-Process -Name GTA5, GTA5_BE, PlayGTAV, GTAVLauncher -ErrorAction SilentlyContinue) { throw 'Feche o GTA V antes.' }

if ($Remover) {
  if (-not (Test-Path $log)) { throw "Sem registro de instalacao em $log" }
  $items = Get-Content $log -Raw -Encoding UTF8 | ConvertFrom-Json
  foreach ($i in $items.files) {
    if (-not (Test-Path -LiteralPath $i.path)) { Write-Host "ja ausente: $($i.path)"; continue }
    if ((Get-FileHash -LiteralPath $i.path -Algorithm SHA256).Hash -ne $i.sha256) { Write-Warning "hash diferente, NAO removido: $($i.path)"; continue }
    Remove-Item -LiteralPath $i.path -Force; Write-Host "removido: $($i.path)"
  }
  foreach ($d in @("$Game\mods\x64\audio\sfx", "$Game\mods\x64\audio", "$Game\mods\x64", "$Game\mods")) {
    if ((Test-Path $d) -and -not (Get-ChildItem $d -Force)) { Remove-Item $d; Write-Host "pasta vazia removida: $d" }
  }
  Rename-Item $log ("instalacao_removida_{0:yyyyMMdd_HHmmss}.json" -f (Get-Date))
  Write-Host 'Remocao concluida.'; return
}

if (Test-Path $log) { throw "Ja instalado (registro $log). Remova antes de reinstalar." }
$copies = @()
# The loader is NOT copied: OpenIV keeps dinput8.dll/OpenIV.asi in its app folder in an encoded form (no MZ header) and only
# writes real DLLs through its ASI Manager. Copying them raw made GTA5.exe exit 0xc000012f (STATUS_INVALID_IMAGE_NOT_MZ).
function Test-PE([string]$p) { if (-not (Test-Path -LiteralPath $p)) { return $false }; $b = [byte[]]::new(2); $s = [IO.File]::OpenRead($p); try { [void]$s.Read($b, 0, 2) } finally { $s.Close() }; return ($b[0] -eq 0x4D -and $b[1] -eq 0x5A) }
foreach ($f in 'dinput8.dll', 'OpenIV.asi') {
  $p = Join-Path $Game $f
  if (-not (Test-Path -LiteralPath $p)) { Write-Warning "$f ausente: instale pelo OpenIV > Tools > ASI Manager (ASI Loader + OpenIV.ASI), senao a pasta mods e ignorada." }
  elseif (-not (Test-PE $p)) { throw "$p nao e uma DLL valida (sem cabecalho MZ); o jogo nao abre assim. Remova e reinstale pelo ASI Manager do OpenIV." }
}
$src = Join-Path $Pacote 'mods\x64\audio\sfx'
foreach ($r in Get-ChildItem $src -Filter *.rpf) {
  $dst = Join-Path "$Game\mods\x64\audio\sfx" (Get-Item -LiteralPath "$Game\x64\audio\sfx\$($r.Name)").Name  # same name/case as the original
  if (Test-Path -LiteralPath $dst) { throw "Ja existe $dst (outro mod?). Nada foi instalado." }
  $copies += [pscustomobject]@{ src = $r.FullName; dst = $dst }
}
foreach ($c in $copies) { if (-not (Test-Path -LiteralPath $c.src)) { throw "Origem ausente: $($c.src)" } }
$done = @()
try {
  foreach ($c in $copies) {
    New-Item -ItemType Directory -Force (Split-Path $c.dst) | Out-Null
    Copy-Item -LiteralPath $c.src -Destination $c.dst
    $done += [pscustomobject]@{ path = $c.dst; sha256 = (Get-FileHash -LiteralPath $c.dst -Algorithm SHA256).Hash; from = $c.src }
    Write-Host "instalado: $($c.dst)"
  }
} finally {
  [pscustomobject]@{ installed = (Get-Date -Format s); game = $Game; files = $done } | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath $log -Encoding UTF8
}
Write-Host "Instalacao concluida. Para desfazer: instalar_teste.ps1 -Pacote `"$Pacote`" -Remover"
