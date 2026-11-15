@echo off
title Fase B cutscenes - NAO FECHAR
powershell -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\gta-dub\work\gta-dialogue\run_cutscenes.ps1' -Phase B -NoLote -SepThreads 8 *> 'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR\cutscenes_lote\logs\phaseB_runner.log'"
