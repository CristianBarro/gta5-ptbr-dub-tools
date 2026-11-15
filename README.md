# gta5-ptbr-dub-tools

Ferramentas experimentais para uma **dublagem PT-BR feita por IA** do GTA V **Legacy** (PC): falas de missão e cutscenes de Michael, Franklin e Trevor.

> **Este repositório contém só código.** Não há áudio do jogo, texto/legendas extraídos, chaves de criptografia, amostras ou modelos de voz. Você precisa da sua própria cópia do jogo. GTA V e todo o seu conteúdo pertencem à Rockstar Games / Take-Two. Projeto de fã, sem fins comerciais e sem afiliação com a Rockstar.
>
> Clonar a voz de uma pessoa real (por exemplo, dos dubladores originais) tem implicações legais e éticas. Use apenas para fins pessoais e de pesquisa e respeite os direitos dos dubladores.
>
> Mods só no **modo história**, com o BattlEye desligado. **Nunca** use mods no GTA Online.

## Estado

Funciona, mas é experimental: nas cutscenes, as falas muitas vezes se embaralham. Há sobreposição, aceleração e atribuição do falante errada em cerca de 12% dos casos. O GTA V **Enhanced** não é suportado, porque usa MP3 nos AWCs (codec 7) e não tem carregador de mods via OpenIV.

## Pipeline

1. **Inventário e texto**
   - `build_catalog.py`, `cut_inventory.py` e `cut_subtitles.py` leem `.cut` e GXT2 e montam a fila de falas.
   - A tradução PT-BR e a fila são geradas fora deste repositório.
2. **Geração (TTS)**: `dub_batch.py`
   - Michael e Franklin: Chatterbox Multilingual (pt), com exagero/cfg por linha para falas gritadas.
   - Trevor: Qwen3-TTS 0.6B em modo x-vector, com `--batch-size` para gerar em lote.
   - O script é retomável e registra cada take em `manifest.json`.
3. **QA e encaixe**
   - Transcrição com Whisper: `dub_qa.py` e `cut_pick.py`.
   - Encaixe de duração: `dub_fit.py`.
4. **Cutscenes**
   - Separação de voz com Demucs htdemucs_ft, só o modelo de vocais: `cut_separate.py`.
   - Quem fala em cada legenda: `cut_speakers_face.py`. Usa a animação facial dos `.ycd` (`YcdProbe`), a regra de vocativo e um veto por voz (CAM++).
   - Remixagem por canal: `cut_mix_batch.py`.
5. **Patch**: `AwcPatcher` (C#, CodeWalker)
   - Reescreve os AWC (IMA-ADPCM, RSXXTEA, AWC multicanal) em cópias dos RPF, numa pasta `mods`.
   - Confere cada stream trocado: correlação ≥ 0,95, e os streams não tocados ficam idênticos.
   - Os arquivos originais do jogo nunca são alterados.
6. **Orquestração e instalação**
   - Orquestração: `run_lote.ps1`, `run_cutscenes.ps1`, `waiter_fase_b.ps1`.
   - Janela de progresso: `status_dub.py`.
   - Instalação e remoção reversíveis, com conferência de hash: `instalar_teste.ps1`.

## Requisitos

- Windows, GPU NVIDIA (testado com 6 GB) e Python 3.10+.
- Pacotes: `torch`, `chatterbox-tts`, `qwen-tts`, `faster-whisper`, `demucs`, `librosa`, `soundfile`, `numpy`.
- .NET 10 SDK. Coloque `CodeWalker.Core.dll`, `SharpDX.dll` e `SharpDX.Mathematics.dll` em `lib/`; eles não estão incluídos.
- Para o jogo ler a pasta `mods`: OpenIV → Tools → ASI Manager → ASI Loader + OpenIV.ASI.

Os caminhos estão fixos nos scripts (`C:\gta-dub`, `D:\JARVIS\...`, pasta do jogo na Steam). Ajuste-os para a sua máquina.

## Lições aprendidas

- No Windows PowerShell 5.1, `$ErrorActionPreference='Stop'` junto com redirecionamento `*>` transforma qualquer linha que um programa nativo escreve no stderr (por exemplo, um `FutureWarning`) em erro fatal.
- O PowerShell não diferencia maiúsculas de minúsculas em variáveis: `$lote` sobrescreve `$Lote`.
- Os falantes nos metadados de conversa das cutscenes estão errados em várias cenas. A animação facial é bem mais confiável.

## Licença

Código sob a licença MIT (veja `LICENSE`). Ela não cobre nenhum conteúdo do jogo nem vozes.
