# Treinos a disparar manualmente (usuário) — dataset data/neoenergia_amr_v2
# Uso: abrir PowerShell na raiz do repo e rodar o bloco desejado.
# Ver: docs/2026-10-19_reimport_neoenergia_det2_uniao.md

# ============================================================
# 1) CRNN — barato, ganho provado (rodar PRIMEIRO)
#    receita do Lab 2 (doc 2026-10-08_cena_det_e_retest_amarelo.md secao 6)
# ============================================================
.venv\Scripts\python.exe -m leiturista.cli train-crnn `
  --data data\neoenergia_amr_v2 `
  --data data\finetune_ufpramr `
  --init models\crnn_bn128_mix500_cur30.pt `
  --invert-prob 0.4 --crop-jitter 0.07 --aug-start 6 `
  --epochs 30 --lr 3e-4 `
  --out models\crnn_dist_v2_ft.pt `
  --experiment crnn-dist-v2

# ============================================================
# 2) TrOCR — leitor principal ponta-a-ponta (rodar DEPOIS)
# ============================================================
.venv\Scripts\python.exe -m leiturista.cli train `
  --model .model_cache\trocr-small-printed `
  --data data\neoenergia_amr_v2 `
  --out models\trocr-small-finetuned-distribuidora-v2 `
  --epochs 8 --batch 4 --grad-accum 8 --lr 2e-5 `
  --experiment trocr-distribuidora-v2

# ============================================================
# 3) Avaliação
# ============================================================
.venv\Scripts\python.exe -m leiturista.cli eval `
  --data data\neoenergia_amr_v2 `
  --out models\trocr-small-finetuned-distribuidora-v2

.venv\Scripts\python.exe -m leiturista.cli compare-crnn `
  models\crnn_lote_ft.pt models\crnn_dist_v2_ft.pt `
  --data data\neoenergia_amr_v2 --split test

# MLflow UI (opcional, para acompanhar as runs):
# .venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
