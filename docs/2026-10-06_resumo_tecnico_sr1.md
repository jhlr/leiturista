# Resumo técnico — entrega do serviço (SR1, BentoML)

Data: 2026-10-06. Escopo: camada de serviço para triagem de fotos de medidor de energia
elétrica. Sem front-end, sem treino, sem deploy em nuvem (conforme o enunciado).

## 1. O que foi entregue

1. **Serviço BentoML** (`src/leiturista/service.py`, BentoML 1.4.39): classe
   `LeituristaService` com um endpoint `POST /predict` (multipart, campo `image`).
   Não tem lógica de OCR própria; envelopa `MeterOCR.predict_image` e traduz a
   `Prediction` para o contrato de resposta.
2. **Contrato de resposta:** `numero_medidor`, `funcao`, `consumo`, `confianca`,
   `legivel`, `flags`.
   - `funcao = sem_leitura_detectada` quando não há leitura.
   - `funcao = leitura_normal` quando há leitura legível e confiança >= 0,5.
   - `funcao = leitura_incerta` nos demais casos (vai para conferência manual).
   - `confianca` é a maior confiança entre os boxes de campo `leitura`; é 0,0 sem
     leitura ou quando a leitura veio do fallback TrOCR na imagem inteira.
3. **Reprodutibilidade:** `pyproject.toml` + `uv.lock`, `.python-version` (3.13),
   `justfile` (`setup`, `models`, `samples`, `serve`, `predict`, `all`),
   `.env.example` (sem segredos), `.gitignore`, `LICENSE` (MIT).
4. **Pesos:** baixados por `just models` de um release público do GitHub
   (`modelos-1.0`, 147 MB), fora do git.
5. **Imagens de exemplo:** sintéticas, geradas por `scripts/gen_sample_images.py`
   (seed fixa, `just samples`).

## 2. Modelo carregado (pergunta 1 da apresentação)

Composição de 3 modelos públicos pré-treinados, nenhum treinado por este grupo:

| Papel | Modelo | Origem |
|---|---|---|
| Detecção de texto | PP-OCRv5_mobile_det (ONNX) | PaddlePaddle, via HF |
| Reconhecimento | PP-OCRv6_tiny_rec (ONNX, 1,1M params) | PaddlePaddle, via HF |
| Fallback de reconhecimento | TrOCR-small-printed (62M) | Microsoft, off-the-shelf |

Fluxo: det acha caixas de texto, rec lê cada crop, cada box é classificado como
`leitura`, `serial` ou `outro`, e flags de coerência são calculadas (legibilidade por
variância do Laplaciano, múltiplas leituras, ausência de texto). Se o det não segmenta
o display, o TrOCR lê a imagem inteira. Há também uma tentativa com imagem invertida
para displays claro-em-escuro.

O fine-tune do TrOCR no UFPR-AMR existe (MLflow), mas o checkpoint usado em produção é
o off-the-shelf.

## 3. Limites (pergunta 2)

Benchmark no UFPR-AMR (300 imagens de teste, display já recortado):

| Modelo | exact-match | digit-acc |
|---|---|---|
| PP-OCRv6_tiny_rec | 0,357 | 0,846 |
| TrOCR-small-printed (fallback) | 0,253 | 0,776 |

1. **A localização do display é o gargalo.** No lote de campo, só ~15% das fotos
   (1.087 de 7.265) tiveram um crop de leitura localizável no import calibrado. As
   heurísticas desse import não estão no `/predict`, então foto de campo crua pode
   devolver lixo (por exemplo, a placa de identificação lida como display).
2. **`funcao` é heurística simples**, não replica o catálogo de notas do cliente
   (casa fechada etc.).
3. **"Sem medidor na foto" e "medidor ilegível" caem no mesmo valor**
   (`sem_leitura_detectada`).
4. A confiança é o score do OCR, não uma probabilidade calibrada contra a decisão
   real de um analista.

Melhorias propostas: detector de visor treinado em foto de campo; classificador
binário "tem medidor?" antes do pipeline; fine-tune do TrOCR com dados de campo;
mapear `funcao` para o catálogo real de notas; calibrar a confiança.

## 4. Como subir (pergunta 3)

```bash
git clone https://github.com/jhlr/leiturista.git && cd leiturista
just all     # uv sync + pesos + imagens sintéticas (~4 min)
just serve   # http://localhost:3000 (Swagger na raiz)
```

## 5. Demonstração (pergunta 4)

```bash
curl -s -F image=@samples/exemplo_01.png http://localhost:3000/predict
```

Casos: leitura normal com alta confiança (`exemplo_01`), leitura + serial
(`exemplo_03`), e caso de erro com foto sem medidor (`sem_leitura_detectada`,
`confianca = 0.0`).

## 6. Uso de IA

Declarado no README (seção 7): ferramenta, pedidos e avaliação crítica.

## 7. Pendências conhecidas na data

1. Dados do cliente rastreados no histórico (CSV de campo e JSONLs de rotulagem) e o
   nome do cliente em docs; exigem reescrita de histórico antes da avaliação.
2. `samples/` não é versionada (as imagens são geradas).
3. Poucos PRs de revisão entre integrantes.
