# Word2Li/Electricity-Meter-OCR-7B — modelo OCR de medidor elétrico

- **URL:** https://huggingface.co/Word2Li/Electricity-Meter-OCR-7B
- **Licença:** MIT
- **Data:** 08/08/2026
- **Status:** download **cancelado** a pedido do usuário (pasta parcial removida).

## O que é

Finetune **end-to-end** do `Qwen/Qwen2.5-VL-7B-Instruct` para ler medidores de
eletricidade direto da imagem, gerando a leitura como texto (ex.: `8430.6`).
8B params, BF16 (≈16 GB), arquivos safetensors, treinado com llama-factory (full).

## Por que existe (versus pipeline clássico)

O card descreve a motivação: métodos tradicionais usam pipeline de 2 estágios
(YOLO pra achar o display + OCR no crop), com fluxo trabalhoso, propagação de erro
da detecção pro OCR e pouca robustez a distorção/luz/ruído. O modelo Image-to-Text
trata a leitura como geração direta e usa contexto global da imagem.

- **Encoder:** ViT (receptive field global) — robusto a rotação, escala e oclusão.
- **Decoder:** LLM autoregressivo — entende estrutura numérica (decimal) e evita
  erros "comuns" de sequência.

## Dados de treino

- **841 imagens anotadas** de medidores elétricos, fornecidas para um **curso** —
  **não publicamente disponíveis** (conjuntos `origin_detection` + `origin_detection_val`).
- 1/10 como validação; 96 acc no test (não disponível).
- Leituras padronizadas em **6 dígitos** (zeros à esquerda/direita preservados).

## Limitações (do card)

- Pode degradar em imagens fora da distribuição: outro tipo de medidor, qualidade
  de imagem ou iluminação diferentes.
- Treinado em dado de curso (provável medidor chinês/Índia) — **não validado em
  medidores BR** (Elster, INMETRO/DIMEL) nem nos nossos datasets locais.

## Como usar (se baixar depois)

```bash
# baixar (≈16 GB, em background sem timeout):
pqenv/bin/hf download Word2Li/Electricity-Meter-OCR-7B --local-dir mapen/Word2Li_Electricity-Meter-OCR-7B

# inferência: pipeline image-to-text (qwen2_5_vl), transformers >= 4.50
```

## Relevância pro MAPEN

1. Referência de abordagem end-to-end — candidato a comparar com pipeline
   local (detecção+OCR) sobre UFPR-AMR (2.000 imgs BR) e os 119 jpgs do usuário.
2. Custo: 8B BF16 local é pesado (≈16 GB + VRAM); alternativa leve no nosso caso
   é o pipeline clássico sobre os ROIs já existentes (`roi_donut`/`roi_counters`).
3. **Sugestão de experimento (potencial de artigo/produto):** finetunar/avaliar
   abordagem end-to-end pequena (Qwen2.5-VL-3B/7B LoRA) no UFPR-AMR — leitura de
   medidor BR com placa metrológica INMETRO é nicho sem modelo público.
