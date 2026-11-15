param(
    [string]$Game = 'D:\SteamLibrary\steamapps\common\Grand Theft Auto V',
    [string]$CodeWalker = 'D:\downloads\1cfcd8-CodeWalker30_dev46\CodeWalker30_dev46',
    [string]$Scripts = 'D:\downloads\GTA-V-Decompiled-Scripts-senpai\GTA-V-Decompiled-Scripts-senpai\decompiled_scripts',
    [string]$Python = 'C:\Program Files\Python310\python.exe'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $taskRoot 'work\bin'
$data = Join-Path $taskRoot 'work\scan'
$result = Join-Path $taskRoot 'rebuild'
New-Item -ItemType Directory -Path $bin -Force | Out-Null
$csc = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$facade = 'C:\Program Files (x86)\Reference Assemblies\Microsoft\Framework\.NETFramework\v4.8\Facades\netstandard.dll'
if (!(Test-Path -LiteralPath $facade)) { $facade = 'C:\Program Files (x86)\Reference Assemblies\Microsoft\Framework\.NETFramework\v4.7.2\Facades\netstandard.dll' }
foreach ($file in @('CodeWalker.Core.dll','SharpDX.dll','SharpDX.Mathematics.dll')) { Copy-Item -LiteralPath (Join-Path $CodeWalker $file) -Destination $bin -Force }
$exe = Join-Path $bin 'ArchiveScanner.exe'
& $csc /nologo /target:exe "/out:$exe" "/reference:$facade" "/reference:$CodeWalker\CodeWalker.Core.dll" "/reference:$CodeWalker\SharpDX.dll" "/reference:$CodeWalker\SharpDX.Mathematics.dll" (Join-Path $PSScriptRoot 'ArchiveScanner.cs')
if ($LASTEXITCODE -ne 0) { throw 'Falha ao compilar o leitor de arquivos.' }
& $exe $Game $data
if ($LASTEXITCODE -ne 0) { throw 'Falha na leitura dos arquivos do jogo.' }
& $Python (Join-Path $PSScriptRoot 'build_catalog.py') --data $data --text-data $data --scripts $Scripts --output $result
if ($LASTEXITCODE -ne 0) { throw 'Falha ao montar o catalogo.' }
Write-Host "Catalogo reconstruido em $result"
