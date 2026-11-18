@echo off
REM CRNN v2 — encoder fine-tune no dataset Neoenergia v2 (uniao de detectores).
REM Basta dar ENTER neste .bat. Receita do Lab 2 (doc 2026-10-08_cena_det_e_retest_amarelo.md secao 6).
cd /d "%~dp0.."

echo === CRNN v2: data\neoenergia_amr_v2 + finetune_ufpramr (ancora) ===
echo encoder: models\crnn_bn128_mix500_cur30.pt  ^| saida: models\crnn_dist_v2_ft.pt
echo.

.venv\Scripts\python.exe -m leiturista.cli train-crnn ^
  --data data\neoenergia_amr_v2 ^
  --data data\finetune_ufpramr ^
  --init models\crnn_bn128_mix500_cur30.pt ^
  --invert-prob 0.4 --crop-jitter 0.07 --aug-start 6 ^
  --epochs 30 --lr 3e-4 ^
  --out models\crnn_dist_v2_ft.pt ^
  --experiment crnn-dist-v2

echo.
echo === terminou. Pressione qualquer tecla para fechar ===
pause >nul
