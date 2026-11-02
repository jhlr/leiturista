# Leiturista

Triagem automática de fotos de leitura de medidor de energia elétrica: a partir de
uma foto, o serviço extrai o número do medidor, classifica se a leitura é confiável
o bastante pra dispensar conferência manual, o valor lido e um grau de confiança.

Projeto da disciplina Projeto 4 / DADOS (Cesar School). O objeto original é uma
distribuidora de energia elétrica parceira da disciplina — o nome dela não aparece
neste repositório por acordo entre a instituição e a empresa; aqui ela é só
"a distribuidora".

**Escopo desta entrega (SR1):** camada de serviço — a inferência exposta atrás de
uma API que outra pessoa/programa consegue chamar. Sem front-end, sem integração
com sistemas do cliente, sem deploy em nuvem. Ver `docs/projeto4_desafio.md` para o
desafio completo do semestre.

## 1. Modelo — o que está sendo carregado

O pipeline (`src/leiturista/inference.py::MeterOCR`) não é um modelo único, é uma
composição de 3 modelos **públicos, pré-treinados** (nenhum foi treinado do zero
por este grupo):

| Papel | Modelo | Origem | Treinado por |
|---|---|---|---|
| Detecção de texto na foto | `PP-OCRv5_mobile_det` (ONNX) | `PaddlePaddle/PP-OCRv5_mobile_det_onnx` (HF) | PaddlePaddle/Baidu |
| Reconhecimento (1ª tentativa, classifica campo `leitura`/`serial`) | `PP-OCRv6_tiny_rec` (ONNX, 1,1M params) | `PaddlePaddle/PP-OCRv6_tiny_rec_onnx` (HF) | PaddlePaddle/Baidu |
| Reconhecimento (fallback, quando o det não segmenta o display) | `TrOCR-small-printed` (62M params, MIT) | Microsoft, off-the-shelf | Microsoft |

Nenhum dos três foi fine-tunado com dados desta entrega — são modelos prontos, o que
o próprio enunciado permite explicitamente ("nenhuma dessas opções vale mais que as
outras nesta nota"). O grupo tentou um fine-tune do TrOCR no dataset público
**UFPR-AMR** (Laroca et al., IJCNN 2020 — `docs/finetune_trocr_ufpramr.md`), mas o
checkpoint salvo em produção hoje (`.model_cache/trocr-small-printed`, MLflow run
`9c14db62`) é o **off-the-shelf**, não o fine-tunado — documentando aqui com
honestidade em vez de inflar a história do modelo.

## 2. Limites do modelo

Medidos no benchmark público UFPR-AMR (300 imagens de teste, `docs/relatorio_benchmark.md`):

| Modelo | exact-match | digit-acc |
|---|---|---|
| PP-OCRv6_tiny_rec (usado no pipeline) | 0.357 | 0.846 |
| TrOCR-small-printed off-the-shelf (fallback) | 0.253 (limpo) | 0.776 (limpo) |

Isso é em fotos de laboratório, já recortadas no display. Em fotos de campo reais
(lote piloto da distribuidora — não versionado, ver seção 6), dois problemas
adicionais aparecem e são maiores que o erro do leitor em si:

1. **Localização do display é o gargalo, não a leitura.** No import batch calibrado
   pro lote real (`docs/fotos_distribuidora_dados.md`), só 1.087 de 7.265 fotos
   (~15%) tiveram um crop de leitura localizável — o resto não achou o display na
   foto (medidor mecânico, ângulo, distância, reflexo). Esse import usa heurísticas
   extras (faixa de dígitos, zeros à esquerda, rejeição de texto de unidade) que
   **não estão** no endpoint genérico `/predict` deste serviço — chamar `/predict`
   direto numa foto de campo real, sem esse recorte prévio, tende a devolver lixo
   (ex.: confundir a placa de identificação do medidor com o display).
2. **O endpoint não distingue "não consegui ler" de "medidor com ocorrência"** (casa
   fechada, leitura informada pelo cliente etc. — os ~61 códigos de nota do cliente,
   `docs/fotos_distribuidora_dados.md`). O campo `funcao` da resposta é uma heurística
   simples (reading + legibilidade + confiança), não replica essas notas.

**Ajustes necessários** pra virar produto: (a) treinar/ajustar um localizador de
display específico pra foto de campo (o gargalo real, por achado próprio), (b)
terminar o fine-tune do TrOCR com dados de campo (autorizado pela disciplina, mas
"treino só o usuário dispara" — não rodado nesta entrega), (c) mapear `funcao` pro
catálogo de notas real do cliente em vez da heurística atual, (d) **classificador
binário no início do pipeline** ("tem medidor nesta foto ou não?", insight do
professor em aula) — hoje o `/predict` roda det+rec+TrOCR completo mesmo numa foto
de portão/fachada/rua; um classificador leve antes disso filtraria essas fotos sem
o custo do pipeline inteiro, e separaria melhor "sem medidor na foto" de "medidor
presente mas ilegível" (hoje os dois caem em `sem_leitura_detectada`).

## 3. Como rodar (do clone à primeira predição)

**Python 3.13** (fixado em `.python-version`; o `pyproject.toml` aceita `>=3.11`, mas só 3.13 foi testado).
O `uv` baixa e usa essa versão sozinho, não precisa instalar Python à mão. Tempo aproximado: ~2 min (`uv sync`
baixa/instala ~160 pacotes) + ~2 min (`just models`, 147 MB) + a primeira chamada de
`bentoml serve` demora ~15-30s pra abrir a porta (import "frio" de torch/transformers)
numa rede razoável.

```bash
git clone https://github.com/jhlr/leiturista.git && cd leiturista
just setup      # uv sync --all-extras — instala as dependências de todos os ambientes (uv.lock)
just models     # baixa + extrai os pesos (release público modelos-1.0, 147 MB)
just samples    # gera samples/*.png (sintéticas, seed fixa — não versionadas)
just serve      # sobe o serviço em http://localhost:3000
```

Sem `just` instalado: `brew install just` (macOS) ou `cargo install just`. Os
comandos equivalentes sem `just`:

```bash
uv sync --all-extras
curl -L https://github.com/jhlr/leiturista/releases/download/modelos-1.0/leiturista-models.tar.gz -o leiturista-models.tar.gz
tar -xzf leiturista-models.tar.gz
uv run python scripts/gen_sample_images.py
uv run bentoml serve leiturista.service:LeituristaService --port 3000
```

O primeiro `predict` demora alguns segundos a mais (carga lazy dos modelos); os
seguintes são rápidos (CPU, sem GPU necessária).

## 4. Contrato da API

**`POST /predict`** — multipart, campo `image` (arquivo de imagem).

```bash
curl -s -F image=@samples/exemplo_01.png http://localhost:3000/predict
```

Resposta:

```json
{
  "numero_medidor": null,
  "funcao": "leitura_normal",
  "consumo": "017355",
  "confianca": 0.987,
  "legivel": true,
  "flags": ["leitura via imagem invertida (display claro-em-escuro)"]
}
```

- `funcao`: `leitura_normal` (confiável, dispensaria conferência manual) |
  `leitura_incerta` (leitura extraída, mas com baixa confiança/legibilidade — manda
  pra conferência) | `sem_leitura_detectada` (não achou display na foto).
- `confianca`: 0.0–1.0, score do bloco de texto reconhecido como leitura. Fica `0.0`
  quando a leitura veio do fallback TrOCR na imagem inteira (sem box localizado).
- Swagger/OpenAPI interativo: `http://localhost:3000/` (a própria raiz do serviço,
  não `/docs` — BentoML 1.4 serve a UI ali).

Casos de teste (`samples/`, todos sintéticos — ver seção 6):

| Imagem | Resultado esperado |
|---|---|
| `exemplo_01.png` | leitura normal, alta confiança |
| `exemplo_03.png` | leitura normal + serial detectado |
| foto sem medidor (qualquer imagem aleatória) | `sem_leitura_detectada`, `confianca=0.0` — caso de erro |

## 5. Como o repo é organizado

```
src/leiturista/
├── inference.py      # pipeline det+rec+TrOCR (MeterOCR) — o "modelo"
├── service.py         # ESTE serviço: envelopa MeterOCR num endpoint BentoML
├── monitoring/        # log JSONL de requisição/resposta, métricas, drift (seção 8)
├── cli.py, data.py, train.py, eval.py, artifacts.py  # pipeline de treino/avaliação (fora do escopo desta nota)
├── distribuidora.py   # import do lote real de campo (dados não versionados)
app/app.py              # demo Streamlit (fora do escopo desta nota — front-end)
scripts/gen_sample_images.py  # gera samples/*.png sintéticos
samples/                # imagens/CSV de exemplo, sem dado real
docs/                   # decisões e achados datados
```

## 6. Dados do cliente

Nenhuma foto ou linha de planilha real da distribuidora está neste repositório —
`FotosDistribuidora/`, `data/`, `models/`, `.model_cache/` e `mlflow.db` são
`.gitignore`d, e o histórico do repo foi conferido (`git log --all --diff-filter=A`)
pra confirmar que nunca entraram. As imagens em `samples/` são **sintéticas**,
geradas por `scripts/gen_sample_images.py` (seed fixa, reprodutível) — não são
versionadas (`samples/*.png` no `.gitignore`), justamente por serem regeráveis em
segundos com `just samples`.

## 7. Uso de IA

Ferramenta: **Claude Code** (Anthropic, Sonnet 5), em modo conversacional, com
acesso a shell, edição de arquivos e GitHub CLI. Detalhado abaixo porque "código
gerado que ninguém da equipe sabe explicar conta como não entregue" — este é o
registro do que foi pedido e a avaliação crítica de cada resposta.

### O que foi pedido, em ordem

1. Revisão de um PR anterior (import do lote real de fotos de campo) — aceitar tudo
   exceto um arquivo de snapshot de sessão que não deveria ir pro repo público.
2. Implementação do SR1 a partir do enunciado do professor, colado inteiro no chat.
3. Três ajustes pontuais durante a implementação: trocar o import do PIL
   (`from PIL import Image as pil`); autorização explícita para testar contra fotos
   reais de campo localmente (sem commitar), em vez de só sintéticas; manter
   `service.py` legível, sem abstração extra em cima do `MeterOCR`.
4. Este relatório de uso de IA, dentro do próprio README.

### O que a IA fez por iniciativa própria (não pedido explicitamente)

- Antes de tocar em qualquer arquivo, auditou se o repositório já era público e se
  continha nome do cliente ou dado real commitado — achou o nome do cliente em 4
  arquivos rastreados (removidos/renomeados) e confirmou via
  `git log --all --diff-filter=A` que nenhuma foto/CSV real tinha sido commitada em
  nenhum momento do histórico.
- Apurou de qual MLflow run vem o TrOCR carregado em produção hoje, cruzando
  `inference.py`, `docs/finetune_trocr_ufpramr.md` e `docs/relatorio_benchmark.md` —
  descobriu que é o checkpoint **off-the-shelf** (`9c14db62`), não o fine-tunado
  como um doc mais antigo sugeria, e reportou isso sem arredondar pra cima.

### Avaliação crítica

**Onde funcionou bem:** o `service.py` gerado é curto (uma classe, um método
`predict`) e foi lido linha a linha antes de aceitar. Todo `curl` deste README foi
de fato executado contra o serviço no ar (imagem pública UFPR-AMR, as 4 sintéticas
de `samples/`, e uma foto real de campo só pra validar a seção de limites, sem
commitar o resultado) — nenhum exemplo é hipotético. Os números de exact-match/
digit-acc citados vieram de doc já existente no repo, não foram inventados.

**Onde precisou de correção humana:** no merge do PR anterior, a IA empurrou a
remoção de um arquivo pra uma branch nova em `origin` em vez de atualizar a branch
do fork de origem do PR — o squash merge trouxe o arquivo de volta pro `main`,
precisando de um segundo commit corretivo depois que o erro foi percebido (registrado
aqui porque é o tipo de erro que passa batido se ninguém conferir o resultado). O
plano inicial também só previa imagens sintéticas pros testes de limite — foi pedido
explicitamente rodar também contra fotos reais de campo antes de escrever a seção 2.
Estilo de import do PIL foi corrigido por pedido direto, não por iniciativa da IA.

## 8. Monitoramento

Cada chamada de `/predict` deixa um rastro em arquivo: uma linha de **requisição** e uma de
**resposta**, ligadas por `request_id`. Monitoramos **variáveis extraídas da imagem** e **variáveis da
saída do modelo**, nunca a imagem, o serial ou o valor da leitura (ver "Privacidade" abaixo). Projeto
técnico completo em `docs/2026-11-01_sdd_monitoramento_mlops.md`.

### Caminho local (sem Docker): do clone a ver o log crescer

```bash
git clone https://github.com/jhlr/leiturista.git && cd leiturista   # 1
just setup                                                          # 2  instala os ambientes (uv.lock)
just models                                                         # 3  pesos dos modelos (147 MB)
just samples                                                        # 4  imagens sintéticas em samples/
just serve                                                          # 5  terminal A: serviço em :3000
just load --images-dir samples --n 20 --rps 1                       # 6  terminal B: gera carga
tail -f serving-logs/inference.log                                  # 7  terminal C: o log crescendo
curl -s localhost:3000/metrics | grep -E "^(bentoml|leiturista)"    # 8  métricas Prometheus
```

`just load` sem argumentos usa o tráfego `holdout` (33 fotos Praekelt reservadas), que exige os
datasets de `data/` (ver `docs/origem_dos_dados.md`); com `--images-dir samples` não precisa baixar nada.
Outras opções: `--traffic henrik|lcd`, `--drift none|mild|strong`, `--invalid-pct`, `--slow-pct`.
O caminho com Docker (`just up-inference`, `just up-monitoring`) ainda não existe nesta etapa.

### Arquitetura e ambientes (alvo)

Treino, inferência e monitoramento são **ambientes separados**, que só compartilham interfaces
explícitas e somente leitura. Nesta etapa existem o ambiente de inferência e o log; o monitoramento
completo (Prometheus, Loki, Grafana) vem na etapa seguinte.

```
 TREINO                      INFERÊNCIA                      MONITORAMENTO
 (existente, intocado)       (docker/inference)              (docker/monitoring)
 grupo uv `train`            imagem `leiturista-inference`   prometheus, loki, alloy,
 CLI `leiturista train`      bentoml serve :3000             grafana, `monitor` (Bento)
        |                    grava log JSONL                       ^        ^
        |  models/ (arquivos) expõe /metrics  ----scrape-----------+        |
        +--montagem ro------> lê models/                                    |
                             ./serving-logs/ (bind mount) --montagem ro-----+
```

| Ambiente | Contém | Não contém |
|---|---|---|
| Treino (`uv sync --extra train`) | pipeline `leiturista`, MLflow, datasets | nada de serviço ou monitoramento |
| Inferência (`uv sync --extra inference`) | biblioteca `leiturista`, BentoML, onnxruntime, torch+transformers, Pillow, numpy, código de log | MLflow, datasets, accelerate, Streamlit, scipy/pandas/matplotlib, qualquer análise de drift |
| Monitoramento (`uv sync --extra monitor`) | BentoML leve, pandas, scipy, matplotlib, httpx | torch, transformers, onnxruntime, pesos |

### O que é gravado (`serving-logs/inference.log`, JSON Lines)

- Evento `request`: `ts` (UTC, microssegundos), `request_id`, `endpoint`, `method`, `service_version`,
  `client_ip` (só um hash de 12 hex), e as variáveis da imagem com prefixo `in_`: `width`, `height`,
  `file_bytes`, `format`, médias de luminância/R/G/B (0-255), histogramas 0-255 de L/R/G/B
  (`in_hist_l`, `in_hist_r`, `in_hist_g`, `in_hist_b`), desvio-padrão, nitidez (variância do Laplaciano),
  canal dominante, fração de pixels estourados/escuros, entropia, densidade de bordas, saturação,
  orientação e faixa dinâmica.
- Evento `response` (sempre, inclusive erros): `status_code`, `latency_ms`, as variáveis da saída com
  prefixo `out_` (`funcao`, `confianca`, `legivel`, `has_reading`, `has_serial`, `reading_digits`,
  `n_boxes`, `n_flags`, `flags`, `candidate_sharpness`; todas nulas se falhou) e `error` (classe da
  exceção + código fixo, sem mensagem).
- Entrada inválida que o BentoML rejeita antes da função da API gera só a linha `response`
  (campo `image` ausente e corpo vazio: 400; arquivo que não é imagem: 500, comportamento do BentoML).
- O arquivo é **append-only**; rotaciona aos 50 MB, com 5 backups (no máximo 300 MB); só `/predict`
  escreve (`/metrics`, `/livez`, `/readyz` não). **Um único worker** (`workers=1`): o handler de
  rotação não é seguro entre processos. `latency_ms` inclui a espera na fila atrás desse worker e, na
  primeira chamada, a carga dos modelos (cerca de 21 s medidos).
- Variáveis de ambiente: `SERVING_LOG_DIR` (padrão `./serving-logs`), `SERVING_LOG_MAX_BYTES`,
  `SERVICE_VERSION`.

**Privacidade:** o log nunca contém a imagem, o número do medidor, o valor da leitura, mensagens de
exceção nem o IP cru. Os nomes da distribuidora não aparecem em nenhum arquivo desta entrega.

### Entendimento dos dados (CRISP-DM)

| Grupo | Variáveis | Tipo | O que descreve |
|---|---|---|---|
| Entrada | `in_luminance_mean`, `in_r_mean`, `in_g_mean`, `in_b_mean`, `in_luminance_std`, `in_sharpness`, `in_width`, `in_height`, `in_file_bytes`, frações de clip, `in_edge_density` | numéricas | condições da foto: brilho, cor, nitidez, tamanho |
| Entrada | `in_format`, `in_dominant_channel`, `in_orientation` | categóricas | formato, dominância de cor, orientação |
| Saída | `out_confianca`, `out_reading_digits`, `out_n_boxes`, `out_n_flags`, `out_candidate_sharpness` | numéricas | comportamento do modelo |
| Saída | `out_funcao`, `out_legivel`, `out_has_reading`, `out_has_serial` | categóricas | proporções de decisão |
| Operação | `latency_ms`, `status_code` | numérica / categórica | saúde do serviço |

1. As variáveis de imagem descrevem as condições da foto (brilho, cor, desfoque, tamanho).
2. As variáveis de saída descrevem o comportamento do modelo (`confianca`, `legivel`, proporção de `funcao`, flags).
3. Uma mudança de iluminação ou de câmera deve mexer primeiro nas entradas e depois nas saídas.
4. `funcao`, `format` e `dominant_channel` são categóricas: acompanham-se por proporção, não por média.
5. Latência e status são variáveis operacionais e se leem à parte das de dados.

### Referência e métricas

- Referência = 132 das 165 fotos Praekelt (divisão determinística, seed 0, `src/leiturista/monitoring/split.py`);
  as outras 33 são o controle "sem drift". `just build-reference` regrava `artifacts/reference.csv` e
  `artifacts/reference_hist.csv` rodando o `MeterOCR` real (cerca de 28 minutos em CPU, porque as fotos
  são grandes). As fotos Praekelt vêm de outro país e de outra câmera que as fotos de campo da
  distribuidora: drift em tráfego real é esperado e é, por si, um achado. Com 132 imagens a referência é
  pequena; leia qualquer KS ou PSI com isso em mente.
- `GET /metrics` (formato Prometheus, embutido no BentoML) traz as séries do BentoML mais três próprias:
  `leiturista_input_luminance_mean`, `leiturista_prediction_confidence` e
  `leiturista_prediction_funcao_total`. Como ler, com um exemplo real:
  `docs/2026-11-01_monitoramento_metrics_analysis.md` e a planilha `.xlsx` com fórmulas vivas.
- Execute os testes do monitoramento com `uv run pytest tests/monitoring` (sobem o serviço de verdade).

## Licença

MIT — ver [`LICENSE`](LICENSE).
