# Reimport Neoenergia com os modelos da release 1.2 (det visor-ft como DET2)

**Data:** 2026-10-19 | **Status:** amostra validada visualmente; run completo disparado
| **Contexto:** após o pull do histórico reescrito de `jhlr/leiturista` (commit `2e9f4a9`) e a
instalação da release **modelos-1.2** (pesos do Lab 2) em `models/`.

## Por quê

O dataset `data/neoenergia_amr` (gerado por `leiturista import-distribuidora` sobre
`data/FotosNeoenergia`, 7.265 fotos `NA`) tinha **6.172/7.265 "sem_crop"** — o gargalo
registrado em `fotos_distribuidora_dados.md` é a **localização do display**, não a leitura.
A release 1.2 trouxe o det PP-OCRv5 fine-tunado no visor (`pp_ocr_v5_mobile_det_visor_onnx`),
plugável no import como 2º detector via `LEITURISTA_DET2_ONNX` (limiar calibrado 0,2/0,2).

## Teste (amostra, 2026-10-19)

Amostra de **500 fotos** (`-n 500`, seed 42 — mesmas fotos nos dois runs) em pastas novas,
mantendo `data/neoenergia_amr` original intacto:

| Config | Saída | aceitos | sem_crop | match ≥ 0,8 | match médio | invertidos |
|---|---|---|---|---|---|---|
| Baseline (det original) | `data/neoenergia_amr_v2_baseline` | 66 | 434 | 43 | 0,855 | 20 |
| **União + visor-ft (DET2)** | `data/neoenergia_amr_v2_uniao` | **87** | 413 | **65** | 0,861 | 43 |

- União: 43 aceitos em comum, **32 novos** (só o det fine-tunado achou o display), 11 que o
  baseline achava e a união perdeu (concorrência de candidatos no `_reading_candidate`, que
  escolhe por match com o rótulo) → **líquido +21 (+32%)**.
- Comando da união (env lida na importação do módulo `inference`, tem que estar setada antes):
  `LEITURISTA_DET2_ONNX=models\pp_ocr_v5_mobile_det_visor_onnx\inference.onnx` +
  `LEITURISTA_DET2_THRESH=0.2` + `LEITURISTA_DET2_BOX_THRESH=0.2`.

### QA visual (mosaicos `_qa/`)

- **Baseline:** 2 erros aparentes nos aceitos (olho do usuário).
- **União:** sem erros aparentes.
- Decisão do usuário: **união** segue como config do reimport.

`LEITURISTA_VISOR_CLS` (classificador de visor) **não afeta este import** — ele só age em
`_best_reading` do pipeline completo; o import escolhe candidato por match vs rótulo.

## Run completo (disparado em 2026-10-19)

```bash
# (PowerShell) LEITURISTA_DET2_ONNX/THRESH/BOX_THRESH como acima
.venv\Scripts\python.exe -m leiturista.cli import-distribuidora ^
  --data-root data\FotosNeoenergia -o data\neoenergia_amr_v2
```

### Resultado final do run completo (2026-10-20)

| status | qtd |
|---|---|
| aceito | **1.479** |
| sem_crop | 5.773 |
| borrado (nitidez < 10) | 12 |
| total processado | 7.264 |

Compare com o import antigo (`data/neoenergia_amr`): **1.087 aceitos / 6.172 sem_crop**.
Ganho de **+392 aceitos (+36%)**. Amostras com `match >= 0,8`: **1.112** (antes: 838).
Mosaicos `_qa/aceitos.png` (40) e `_qa/rejeitados.png` (12) conferidos visualmente pelo
usuário — **sem erros aparentes**.

Split (por lote `LOTE_SPLIT`, cronológico): **train 745 / valid 358 / test 376**.
Nota: a proporção não é 80/10/10 porque é a partição por lote, não aleatória — o lote de
valid (210526) e o de test (030726) renderam mais crops que no import antigo
(antes: 869 / 108 / 110).

Run anterior travou no meio (237 crops) por queda da sessão do terminal;
`scripts/run_import_v2.ps1` roda o import desacoplado (`Start-Process`) com log em
`data/neoenergia_amr_v2/import.log`, sobrevivendo à sessão.

## Próximo passo (fine-tune — só o usuário dispara)

Com `data/neoenergia_amr_v2` pronto:

- **TrOCR:** `leiturista train --model .model_cache/trocr-small-printed --data data/neoenergia_amr_v2 ...`
- **CRNN:** `leiturista train-crnn --data data/neoenergia_amr_v2 --init models/crnn_bn128_mix500_cur30.pt ...`
  (receita do Lab 2, doc `2026-10-08_cena_det_e_retest_amarelo.md` seção 6)
- Avaliar em seguida com `leiturista eval` / `compare-crnn`.

## Fine-tune das duas famílias de leitor (2026-10-20)

Scripts prontos: `scripts/treino_crnn_v2.bat`, `scripts/treino_trocr_v2.bat`,
`scripts/treinos_v2.ps1`. Env do 2º detector vive em `scripts/run_import_v2.ps1`.

### CRNN (`crnn_dist_v2_ft.pt`) — receita do Lab 2

`--data data/neoenergia_amr_v2 --data data/finetune_ufpramr --init
models/crnn_bn128_mix500_cur30.pt --invert-prob 0.4 --crop-jitter 0.07 --aug-start 6
--epochs 30 --lr 3e-4`. Encode com `n_train = 751` (crops do v2 + âncora UFPR).

| Métrica | Valor |
|---|---|
| valid exact | 0,538 |
| valid digit-acc | 0,835 |
| `neoenergia_amr_v2` valid exact | 0,573 |
| `neoenergia_amr_v2` **test exact** | **0,521** |
| `neoenergia_amr_v2` test digit-acc | 0,819 |
| `finetune_ufpramr` test exact | 0,500 (âncora preservada, test digit-acc 0,862) |

`compare-crnn` no mesmo test (n = 376):

| Modelo | Leitura exata |
|---|---|
| `crnn_lote_ft.pt` (Lab 2, dataset antigo) | 0,399 |
| **`crnn_dist_v2_ft.pt` (dataset v2)** | **0,521** |

McNemar pareado: só o antigo acerta 13, só o novo acerta 59, **p = 0,000**. O dataset
v2 é significativamente melhor — coerente com os +392 recortes e o match ≥ 0,8 maior.

### BUG 1: TrOCR falhava sempre no Windows (collator não-serializável)

`src/leiturista/train.py` usava `collator = lambda batch: _collate(batch, tokenizer)`.
Com `dataloader_num_workers=2`, o `DataLoader` precisa **picklar** o collator para os
subprocessos; um lambda definido dentro de `train()` **não é serializável**, e no Windows
(spawn) o treino morria em ~1 s com `_pickle.PicklingError: Can't pickle local object
<function train.<locals>.<lambda>>`. Corrigido substituindo o lambda por uma classe
`_Collator` de nível de módulo (chamável e picklável). Validado com smoke test de 1 época
ponta-a-ponta.

(Os avisos `missing keys`/`unexpected keys` que o `transformers` imprime ao carregar o
checkpoint são ruído de nomenclatura de camadas, não erro.)

### BUG 2: `pickle data was truncated` com os workers do DataLoader

Corrigido o collator, o TrOCR passou a morrer em ~4 s com
`_pickle.UnpicklingError: pickle data was truncated`, na serialização do **Dataset Arrow**
para os subprocessos. No Windows (`spawn`) os workers precisam picklar o dataset inteiro
pelo pipe do multiprocessing, e ele estoura. Corrigido com
`dataloader_num_workers=0` (`train.py`): carrega no processo principal; o I/O de PNG é
rápido e o gargalo é o modelo, não o disco.

### BUG 3: `OverflowError` na avaliação (`_exact_match`)

Com os dois fixes anteriores, o treino passou a rodar e **morreu no passo 50 — o primeiro
`eval`**:

```
File "src/leiturista/train.py", line 54, in _exact_match
OverflowError: can't convert negative int to unsigned
```

Sintoma enganoso: parecia falha de treino (o processo morria no meio e o exit code era 1),
mas o treino estava ótimo — quebrava só na primeira avaliação.

**Causa raiz:** o tokenizer do TrOCR (`XLMRobertaTokenizer`, backend `tokenizers`/rust)
estoura com `OverflowError` quando recebe **qualquer id negativo**. Não é questão de dtype
nem de `uint`: passando lista de Python, `ndarray` ou tensor, o erro é o mesmo. Os negativos
vêm do `generate` (`-1`) e do padding do collator (`-100`). A `_decode` do `transformers`
repassa a sequência crua ao `tokenizers`, e a ponte com o rust não aceita valores negativos.

**Fix:** filtrar os ids negativos **antes** de decodificar (`_decode_ids`, `train.py`).
Eles não representam caractere, então descartá-los não muda a leitura — verificado:
`decode([100,200,30,5,6,-1]) == decode([100,200,30,5,6])`.

**Como cheguei na causa (e o que deu errado antes):** tentei três fixes no lugar errado —
converter para `int64`, normalizar no `compute_metrics`, usar `.tolist()`. Todos falharam
porque eu estava tratando o **sintoma** (dtype/formato do array) sem nunca reproduzir o caso
real. Foi só quando parei de relançar o treino de 30 min e escrevi um teste que rodava o
`eval` do `Trainer` de verdade, imprimindo os dtypes, que vi que o array já chegava `int64`
e `-1`/`-100` (não `uint`) era o gatilho. Lição registrada: **reproduzir antes de corrigir**.

Validado com teste isolado cobrindo 6 formatos (`ndarray` 1-D/2-D `int64`/`uint32`, tupla,
lista de listas, tensor torch), todos com `exact_match = 1.0` e sem `OverflowError`.

Junto com os bugs 1 e 2, o `leiturista train` **não funcionava no Windows** desde o
desmembramento do `mapen` (que rodava em macOS): os três são exclusivos do Windows
(spawn/pickling/comportamento do tokenizer).

### BUG 4: `sqlite3.DataError: string or blob too big` ao salvar o `checkpoint.zip`

Com o BUG 3 corrigido, o treino **completou as 3 épocas** e morreu no último passo, salvando
o artefato:

```
File "src/leiturista/artifacts.py", line 46, in save_blob
sqlite3.DataError: string or blob too big
```

Causa: artefatos são blobs num tabela do `mlflow.db`, e o SQLite tem limite de **1 GiB**
(`SQLITE_LIMIT_LENGTH = 1.000.000.000`, medido nesta máquina). O diretório de saída do
TrOCR somava **1.736,6 MB**:

| Arquivo | Tamanho | Presença |
|---|---|---|
| `optimizer.pt` (×2 checkpoints) | 491,9 MB cada | estado transitório de treino |
| `model.safetensors` (×3) | 246,4 MB cada | **é o modelo** |
| `tokenizer.json` (×3) | 4,5 MB cada | necessário |

O `save_total_limit=2` duplica o otimizador: dois arquivos de ~490 MB sozinhos já estouram.

**Fix:** `artifacts.save_zip` ganhou `exclude` (casa pelo **basename**) e `exclude_glob`
(caminho relativo via `fnmatch`); `train.py` exclui `optimizer.pt`, `scheduler.pt`,
`rng_state.pth` e `training_args.bin`. Resultado medido: **1.736,6 MB → 752,8 MB**, com
`model.safetensors` e os configs preservados. O otimizador é estado transitório — perdê-lo
custa apenas a retomada exata do treino, não a inferência nem o fine-tune a partir dos pesos.

Detalhe que me custou tempo: a primeira versão do filtro casava por
`rel.startswith(exclude)`, com `rel = p.relative_to(src).as_posix()`. Isso **nunca casa em
arquivo dentro de subpasta** — `"checkpoint-50/optimizer.pt".startswith("optimizer.pt")` é
`False`, e `== "optimizer.pt"` também. O padrão `"checkpoint-*/optimizer.pt"` não ajuda,
porque `startswith` não interpreta glob. Corrigido para casar pelo `Path(p).name`.
Validado chamando a `save_zip` **real** (não uma cópia da lógica) contra o checkpoint em
disco, gravando num `run_id` descartável e apagando o blob no fim.

### BUG 5 (meu): perdi o run do TrOCR apagando runs "mortos"

Na limpeza dos runs que falharam, usei o filtro `status IN ('FAILED', 'RUNNING')` esperando
poupar o run `FINISHED` com as métricas reais. **O experimento ficou vazio:** o run FINISHED
foi apagado junto. Causa: o run original tinha nome `train-3ep` — o mesmo nome dos 5 runs
`FAILED` da mesma série. A consulta de depuração devolveu 9 runs em vez de 8, e eu não parei
para conferir a lista de alvos antes de executar.

Dados recuperados do `trainer_state.json` que o `Trainer` deixa em disco (`checkpoint-72`):
as 38 métricas foram reinseridas e o run marcado com `reconstruido=true` + `motivo`
explicando a origem. Nada se perdeu de verdade — o `trainer_state.json` guarda o
`log_history` completo, com treino e avaliações por passo.

Lição: em script destrutivo, **imprimir a lista de alvos antes** de apagar, não só a
contagem depois.

### Limitação de hardware: sem GPU nesta máquina (2026-10-20)

Medido no ambiente do projeto:

| Item | Valor |
|---|---|
| `torch.__version__` | **2.14.0+cpu** (build só-CPU, sem CUDA) |
| `torch.cuda.is_available()` | `False` (`device_count = 0`) |
| GPU do sistema | **AMD Radeon (TM) Graphics** integrada, ~512 MB |
| `nvidia-smi` | não existe (sem GPU NVIDIA) |

Consequências:

1. **TrOCR só roda em CPU**: batch 4 × grad-accum 8 dava **~22 s/passo** →
   ~1h11/época → **~9,5 h** para 8 épocas. Ajustado para **3 épocas** com
   `--batch 8 --grad-accum 4` (mesmo produto de gradiente = 32, menos overhead de passo),
   estimativa ~2,5–3 h. Sem 8 épocas por restrição de hardware, não por escolha.
2. O CRNN é bem menos afetado (modelo pequeno, 32×128): 30 épocas em CPU são viáveis.
3. `pyproject.toml` declara `torch` sem índice CUDA — numa máquina com GPU NVIDIA seria
   preciso instalar o build `cu*` (`--index-url .../whl/cu124`) e o `models.resolve_device`
   passaria a usar CUDA sozinho.
4. DirectML (torch-directml) foi considerado para a Radeon integrada e **não** adotado:
   ganho incerto num modelo de ~250M e risco de incompatibilidade com o `Trainer`.

### TrOCR v2: treino concluído, resultado ruim

`scripts/treino_trocr_v2.bat` → `models/trocr-small-finetuned-distribuidora-v2`, 3 épocas
(`--batch 8 --grad-accum 4`, ver limitação de hardware acima), lr 2e-5, experimento
`trocr-distribuidora-v2`. Treino completou: 72 passos, `train_runtime` 1.869 s (~31 min),
`train_loss` 18,8 (queda limpa: 46 → 7,0).

Avaliações no valid durante o treino (`exact_match` por passo):

| Passo | Época | `eval_loss` | `eval_exact_match` |
|---|---|---|---|
| 50 | 2,09 | 2,000 | **0,1480** ← melhor |
| 72 | 3,00 | 1,758 | 0,1313 |

O `load_best_model_at_end=True` restaurou o checkpoint-50 (época 2) em `trainer.model`, e
foi ele que ficou gravado na raiz do diretório de saída.

**Resultado no test completo (n = 376, mesmo split do CRNN):**

| Modelo | Leitura exata (test v2) |
|---|---|
| CRNN `crnn_dist_v2_ft.pt` | **0,521** |
| TrOCR v2 (raiz, época 2) | 0,168 |
| TrOCR v2 (`checkpoint-50`) | 0,130 |

O TrOCR fine-tunado é **~3,1× pior que o CRNN** (0,168 vs 0,521) e perde da própria loss de
treino, que seguia caindo na época 3: `eval_loss` melhora (2,00 → 1,76) enquanto
`exact_match` **piora** (0,148 → 0,131). É o padrão clássico de **overfitting em leitura
parcial** — o modelo aprende a acertar mais *tokens* (loss menor) errando o número completo,
e a leitura exata cai. A leitura de 3 amostras ilustra o modo de falha:

```
pred='48887' label='48807'   # 1 dígito trocado
pred=''      label='7076'    # não gerou nada
pred='488'   label='48807'   # truncou no meio
```

Sinais de que o modelo (~250M) é grande demais para 745 amostras de treino: `grad_norm`
oscilando entre 89 e 208 sem convergir, e um dígito errado/truncamento em vez de erro
grosseiro.

Conclusão prática: **o CRNN é o leitor a usar** neste dataset. O TrOCR precisaria de muito
mais dado (ou `trocr-base` com encoder congelado, ou currículo por tamanho de número) para
competir. Não vale gastar ~31 min de CPU por rodada em mais épocas enquanto não houver mais
rótulos — o gargalo é o dataset, não o número de épocas.

Sobre a raiz vs `checkpoint-50`: a raiz pontua melhor (0,168 vs 0,130), mas os pesos não são
idênticos a nenhum checkpoint em disco, porque o `load_best_model_at_end=True` recarrega o
melhor estado em `trainer.model`. Como o "melhor" foi escolhido pelo **valid** e o test
discorda, isso é sobretudo ruído de seleção com amostras pequenas — mais um motivo para
tratar o TrOCR como não-competitivo aqui.

**O TrOCR fine-tunado é ~3,5× pior que o CRNN** e perde da própria loss de treino (que
seguia caindo na época 3): `eval_loss` melhora (2,00 → 1,76) enquanto `exact_match` **piora**
(0,148 → 0,131). É o comportamento clássico de **overfitting em leitura parcial** — o modelo
aprende a acertar mais *tokens* (loss menor) errando o número completo, e a leitura exata
cai. Sinais de que o modelo é grande demais para o dataset (745 amostras de treino):
`grad_norm` oscilando entre 89 e 208, sem convergir.

Conclusão prática: **o CRNN é o leitor a usar** neste dataset. O TrOCR precisaria de muito
mais dado (ou um `trocr-base` com congelamento do encoder, ou currículo por tamanho de
número) para competir. Não vale gastar ~30 min/rodada de CPU em mais épocas enquanto não
houver mais rótulos — o gargalo é o dataset, não o número de épocas.

