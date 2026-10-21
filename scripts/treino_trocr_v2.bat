@echo off
REM TrOCR v2 — leitor principal no dataset Neoenergia v2 (rodar DEPOIS do CRNN).
cd /d "%~dp0.."

set MLFLOW_DISABLE_AGENT_HINT=1
echo === TrOCR v2: data\neoenergia_amr_v2 ===
echo base: .model_cache\trocr-small-printed  ^| saida: models\trocr-small-finetuned-distribuidora-v2
echo nota: CPU-only nesta maquina (torch +cpu, sem GPU NVIDIA) -> 3 epocas, batch 8 x grad-accum 4
echo.

.venv\Scripts\python.exe -m leiturista.cli train ^
  --model .model_cache\trocr-small-printed ^
  --data data\neoenergia_amr_v2 ^
  --out models\trocr-small-finetuned-distribuidora-v2 ^
  --epochs 3 --batch 8 --grad-accum 4 --lr 2e-5 ^
  --experiment trocr-distribuidora-v2 > logs\trocr_v2.log 2>&1

echo.
echo === terminou (exit=%ERRORLEVEL%). Log: logs\trocr_v2.log ===
pause >nul
