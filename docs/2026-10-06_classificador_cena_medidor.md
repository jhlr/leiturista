# Classificador de cena: "tem medidor na foto?"

**Data:** 2026-10-06 · Estágio 1 da triagem (ver `2026-10-01_gargalo_amarelo_e_plano_visor.md`).
Converte "sem medidor" (~8% das fotos) de amarelo em vermelho antes de gastar OCR.

## O que é

`src/leiturista/scene.py`: MobileNetV3-Small (ImageNet) com 1 logit = P(tem medidor), foto
inteira em 320x240. Backbone congelado nas 2 primeiras épocas. BCE ponderada pela classe rara.
Augmentation: rotação 0/90/180/270, flip, jitter de cor, blur. MLflow em `mlflow.db`
(experimento `scene-medidor`).

## Dados e limites

1. Rótulos: `llm_meter_visible` do CSV da distribuidora (3.247 fotos; **rótulo fraco**, feito por
   LLM). 2.978 com medidor, 269 sem. Split 80/10/10 estratificado, seed 42:
   train 2.597 (215 neg), valid 323 (26 neg), test 327 (28 neg).
2. Só 26 e 28 negativos em valid/test: as métricas de "sem medidor" têm barra de erro grande.
   Antes de confiar no número, conferir à mão os negativos do test.
3. O limiar sai do `valid` (maior F1 da classe "sem medidor") e fica no checkpoint.
   O `test` só reporta; não seleciona nada.
4. O CSV com os rótulos e as fotos ficam só em disco local (`data/`, gitignored):
   `data/distribuidora_campo_rotulado.csv` e `data/distribuidora_campo/`.

## Uso (treino é do usuário)

```bash
caffeinate -i .venv/bin/leiturista train-scene          # salva models/scene_medidor.pt
```

Métricas reportadas: AUROC (valid/test) e precisão/recall/F1 da classe "sem medidor".
Integração no `/predict` (`predict_scene`) fica para depois de ver os números.
