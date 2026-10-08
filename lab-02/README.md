# Lab 2 — Luminus: respostas, tabelas e leitura dos resultados

Bloco profundo: **`CRNNDigitos`** (CNN + GRU bidirecional + CTC), o reconhecedor de dígitos do display recortado.
Notebook executado do zero: [`lab02.ipynb`](lab02.ipynb). Ficha: [`ARQUITETURA.md`](ARQUITETURA.md). Rótulos:
[`rotulos/`](rotulos/). Código na biblioteca `leiturista` (`src/leiturista/`), números em `docs/figs/*.json`.

## 0. Discordância da leitura do docente (com evidência)

O enunciado diz "tudo off-the-shelf; benchmark só em dado público". Isso já mudou: o PP-OCR e o TrOCR continuam
off-the-shelf no pipeline de produção, mas o bloco profundo do grupo é um **CRNN próprio, ajustado em recortes do
cliente** e medido em lote nunca visto (L2). O que **permanece verdadeiro**: o score de coerência não está calibrado
contra uma decisão real, e a rotulagem humana (item C) depende do grupo (ver §C, estado honesto).

## B. O modelo profundo em código

- **B1 (tabela de formas, entrada 1×32×128):** `cnn.0 (32×16×64) → cnn.1 (64×8×32) → cnn.2 (128×4×32) → cnn.3 (128×1×32) →
  rnn (32×256) → saída (32×11)`. A largura é o eixo de tempo (T = 32).
- **B2 (parâmetros):** 441.931 no total. Ajuste fino: 441.931 treináveis (100%). Extração (CNN congelada): 200.971 treináveis (45%).
- **B3 (ativação e perda):** `Linear(256→11)` por passo, perda `CTCLoss(blank=0)` (aplica `log_softmax`; **sem softmax no modelo**).
  Por passo as 11 classes (10 dígitos + branco) são **exclusivas** (softmax); sigmoid por dígito não diria onde nem
  quantas vezes cada dígito aparece. Prova na base: 795 de 2.547 leituras reais têm dígito repetido em sequência
  (`81999`, `5599`, `44248`); sem o branco da CTC, `99` colapsaria em `9`.

## C. Rótulos e partição

- **C1:** [`rotulos/esquema.md`](rotulos/esquema.md): classes `legivel`/`ilegivel`/`sem_medidor`, `leitura` transcrita
  quando legível, regras de desempate e 2 exemplos-limite por classe.
- **C2/C3 (estado honesto):** o plano está pronto (300 fotos, 75 por lote; 50 em dupla às cegas, rotuladores diferentes,
  `rotulos/plano.csv`), o app de rotulagem roda (`streamlit run app/rotular.py`, cada integrante vê só o seu bloco) e o
  kappa é calculado por `leiturista rotulos-kappa`. **O plano de 300 fotos por 4 lotes não foi cumprido**, mas há duas fontes
  de rótulo de classe que o grupo passou a usar:
  1. **Humano (Mikael):** 3.063 fotos de **um só lote** (`220526_0408`), triadas em pastas e convertidas ao esquema
     (`medidor`→`legivel` 610, `medidoruim`→`ilegivel` 2.187, `ntem`→`sem_medidor` 266), em
     [`rotulos/rotulos_mikael.csv`](rotulos/rotulos_mikael.csv). Um rotulador, **sem leitura transcrita e sem dupla**.
  2. **Modelo de captioning local (rótulo fraco; colunas `llm_*`):** 3.247 fotos de campo com `llm_meter_visible`/`llm_display_readable` em
     `data/distribuidora_campo_rotulado.csv`, convertidas pela regra: medidor não visível = `sem_medidor`; visível e display
     legível = `legivel`; visível e ilegível = `ilegivel`.
  - **C3 (concordância entre rotuladores), feito com humano × modelo de captioning local.** O enunciado pede dois integrantes, às cegas, em 50
    fotos. O grupo não tinha um segundo humano; usou como segundo rotulador um **modelo de captioning (visão-linguagem) rodando localmente**
    (Llama-3.2-Vision 11B), cujas descrições das fotos (colunas `llm_*`) foram convertidas em classe antes do rótulo do Mikael existir (não viu a resposta
    dele, então a independência vale; o Mikael também não viu a do LLM). A amostra é maior que a pedida: **688 fotos** rotuladas
    pelos dois, todas do lote `220526_0408`. Não é dupla humana e isso muda o que o número prova: mede o quanto o LLM
    concorda com um humano, não o quanto dois humanos concordam sobre a definição das classes.

    | | Concordância | Kappa de Cohen |
    |---|---|---|
    | Geral (3 classes) | 0,562 | 0,324 |
    | `legivel` (um contra todos) | | 0,285 |
    | `ilegivel` | | 0,239 |
    | `sem_medidor` | | 0,778 |

    Como o enunciado manda, kappa < 0,6 em `legivel` e `ilegivel` indica definição ambígua; o esquema foi revisado
    (seção "O que mudou depois da medição de concordância" em `rotulos/esquema.md`, regras 5 e 6 novas). O desacordo é quase
    todo de um tipo: em 276 das 688 fotos o modelo disse legível e o humano ilegível (o oposto ocorre em 1); o modelo marca 61%
    como legíveis, o humano 21%. Auditoria visual de 12 dessas 276 (miniaturas, a olho): 5 são ilegíveis de fato (grade, vidro
    trincado, display minúsculo, reflexo; o modelo foi leniente), 2 borderline e 5 parecem legíveis (o humano foi mais
    rígido, ou só em resolução cheia os dígitos exigiriam chute). **Os erros vão nos dois sentidos**, e a amostra pequena não
    permite dizer qual pesa mais. `sem_medidor` tem concordância boa (0,78). Cálculo: `leiturista.rotulos.cohen_kappa`, mesma
    função do `leiturista rotulos-kappa`. **Não feito por falta de tempo:** a dupla humano × humano em 50 fotos (o que o
    enunciado pede em C3 e o que decidiria o desacordo acima); o app de rotulagem e o `plano.csv` estão prontos para isso.
  - **Arquivo final:** [`rotulos/rotulos.csv`](rotulos/rotulos.csv) (5.622 fotos; colunas e fontes em `rotulos/esquema.md`). A `leitura` é a digitada pelo leiturista em campo (`Posicao do medidor lida`, 559 das 610 fotos `legivel` do Mikael têm uma), **não conferida na foto**: o L2 abaixo é, por isso, rótulo fraco de leitura, com classe de legibilidade humana.
  - **Regra de união** ([`rotulos/rotulos_uniao.csv`](rotulos/rotulos_uniao.csv), 5.622 fotos): `legivel` se pelo menos uma das
    fontes disser legível. Nas 688 comuns dá 419 legíveis, 222 ilegíveis, 47 sem medidor. Escolha de recall ("vale tentar ler"),
    não de certeza: contraria a regra 1 do `esquema.md` (dígito duvidoso torna a foto ilegível) e puxa para o erro caro (falso
    verde). As colunas humano e LLM ficam separadas no arquivo para trocar a regra sem refazer nada.
  Enquanto isso, os itens D, E e L abaixo usam como proxy os **recortes de display do cliente** (1.091 recortes
  por lote, rótulo = leitura digitada pelo leiturista confirmada pelo OCR: rótulo fraco, enviesado para casos fáceis).
- **C4 (partição por lote, sem vazamento):** treino = lotes `200526_0352` e `210526_0335` (547 recortes), valid =
  `220526_0408` (264), teste = `030726_0121` (280). Interseção de medidores, fotos e imagens entre conjuntos: **0 em todos
  os pares** (saída em `lab02.ipynb`, função `split_by_lote`).

## D. Sanidade e baseline

- **D1:** perda CTC inicial com pesos aleatórios = 14,3 num lote real de 16 recortes; referência da saída uniforme sobre 11
  símbolos, T·ln(11)/L = 32·ln(11)/L̄ = 17,05 (a CTC divide pelo comprimento do alvo). Mesma ordem de grandeza: inicialização sã.
- **D2:** 16 recortes **reais**, 300 passos sem regularização: perda 14,33 → 0,004, leitura exata 16/16.
- **D3 (3 sementes, mesma partição por lote, mesma métrica = leitura exata, mesmos dados de treino):**

| Modelo | valid | teste |
|---|---|---|
| Baseline não profunda: HOG + regressão logística por posição (alpha escolhido no valid) | 0,000 ± 0,000 | 0,000 ± 0,000 |
| CRNN, modo extração (CNN congelada, cabeça GRU+Linear reiniciada) | 0,381 ± 0,009 | 0,376 ± 0,011 |
| CRNN, ajuste fino completo (1 treino; `crnn_lote_ft`) | 0,553 | 0,521 |

  O profundo ganha de longe. A baseline não profunda erra tudo porque 5 posições independentes sobre HOG não aprendem o
  alinhamento da sequência (treino 100% por posição, valid ~20%): memoriza e não generaliza. Ressalva: uma baseline
  mais forte (segmentação de dígitos + SVM) não foi tentada.

## E. Confiança e recusa

Confiança de uma leitura = média geométrica das probabilidades máximas por passo da CTC, com temperatura T nos logits.
**T = 3,76 ajustado só no valid** (264 recortes).

| | ECE (teste, 10 bins) |
|---|---|
| Antes (T = 1) | 0,432 |
| Depois (T = 3,76) | 0,193 |

Figura: [`docs/figs/e1_e2_calibracao_cobertura_risco.png`](../docs/figs/e1_e2_calibracao_cobertura_risco.png).
O CRNN era superconfiante; a calibração ajuda, mas 0,19 ainda é alto (a confiança ordena bem, AUROC 0,86 contra acerto,
mas não é bem calibrada em valor).

**E2, meta do kickoff (amarelo < 40% do volume ⇒ decidir ≥ 60% das fotos) escrita como "cobertura ≥ 60% com risco ≤ 2%"
(risco = leitura errada entre as aceitas; é o falso-verde, o erro caro):** **não cabe na curva.** Com o CRNN atual:

| Risco máximo | Cobertura máxima |
|---|---|
| ≤ 2% | 0,4% |
| ≤ 10% | 4,3% |
| ≤ 20% | 39,6% |
| (cobertura 60%) | risco 26,8% |

Leitura: com leitura exata de 52% no lote novo, não dá para aceitar 60% das leituras com 2% de erro só pela confiança do
reconhecedor. A meta só é atingível combinando reconhecedor melhor (mais recortes reais), a regra de coerência com a nota e a
política de foto ilegível (R6, vermelho), não pela recusa isolada.

## L. Parte específica — Luminus

- **L1 (como cada família erra, 150 recortes do cliente no lote de teste; 10 erros reais de cada em `docs/figs/l1_erros.json`):**

| Família | Acerto exato | Trocou | Perdeu | Duplicou | Inventou |
|---|---|---|---|---|---|
| PP-OCRv6 tiny (CRNN+CTC, off-the-shelf) | 0,253 | 35 | 0 | 10 | 67 |
| TrOCR-small (ViT + Transformer) | 0,080 | 13 | 37 | 5 | 83 |
| CRNNDigitos fine-tunado (nosso) | 0,547 | 24 | 15 | 1 | 28 |

  Critério (alinhamento de edição, `leiturista.errors.classify_error`): *perdeu* = só deleções; *trocou* = só substituições;
  *duplicou* = inserção de dígito igual a um vizinho; *inventou* = o resto (inclui inserir um dígito novo, ex.: `02019` por `2019`).
  Entrada/saída/perda/parâmetros: CRNN+CTC = imagem 1×32×128 (ou 3×48×W no PP-OCR), saída T×classes, CTC, 0,44M (nosso) e 1,1M
  (PP-OCRv6 tiny); TrOCR = patches ViT, tokens autoregressivos, entropia cruzada por token, ~62M. Como erram: a CTC tende a
  trocar, perder ou inserir um dígito isolado; o TrOCR **inventa** ou devolve vazio (83 invenções, 37 vazios em 138 erros),
  o modo de falha que o guia aponta como mais perigoso para dígito de fatura.

- **L2 (benchmark do cliente × UFPR-AMR; lote de teste nunca visto, 280 recortes do cliente e 300 do UFPR-AMR):**

| Modelo | Cliente (exata / por dígito) | UFPR-AMR (exata / por dígito) | Gap (exata) |
|---|---|---|---|
| PP-OCRv6 tiny | 0,261 / 0,746 | 0,403 / 0,814 | 0,14 |
| TrOCR-small | 0,075 / 0,400 | 0,373 / 0,785 | 0,30 |
| CRNN só UFPR-AMR (zero-shot) | 0,039 / 0,381 | 0,873 / 0,945 | 0,83 |
| **CRNN fine-tunado no cliente (por lote)** | **0,521 / 0,823** | **0,907 / 0,960** | 0,39 |

  O gap de domínio é enorme para o CRNN treinado só em público (0,87 → 0,04) e o ajuste fino em 547 recortes do cliente recupera
  0,52. Ressalvas: rótulos fracos e enviesados; ≥ 200 recortes humanamente transcritos virão de C2. Ajuste fino por lote
  derrubou o número de 0,609 (partição aleatória por foto) para 0,521: a partição aleatória inflava.

- **L3 (métrica declarada):** **otimizadora = leitura exata** (a fatura sai errada com um dígito trocado; por dígito esconde
  isso). Restrições satisfatórias: (1) **risco ≤ 2%** entre as leituras aceitas como verde (falso-verde); (2) **latência p90 ≤ 2 s
  por foto em CPU**, hoje 11,2 s no pipeline atual (ver L4) e ~0,6 s estimados com o CRNN como leitor.
- **L4 (plano de ajuste fino e latência):**
  - *Congelar:* nada no ajuste fino principal (a CNN já foi treinada na mesma tarefa no UFPR-AMR; congelar custou 14 pontos
    de leitura exata no D3: 0,376 × 0,521). *Taxas:* 3e-4 (Adam), warmup 50 passos, clipping calibrado no p90 da norma.
  - *Aumento permitido:* brilho/contraste, giro ±10°, desfoque leve, inversão de polaridade (41,7% dos recortes vêm da fase
    invertida), jitter de recorte ±7%, glare/oclusão por dígito. *Proibido para dígitos:* espelhamento (dígito espelhado não
    existe), rotação grande, deformação elástica. *Descoberta:* ligar toda a augmentation desde a época 0 travou a CTC no
    platô (só emite branco); resolveu-se com **currículo** (6 primeiras épocas sem augmentation).
  - *Onde está a latência (60 fotos, CPU, p90 total 11,2 s):* **TrOCR = 93% do tempo** (média 3,7 s, p90 10,8 s); detecção 0,27 s;
    correção de rotação 0,17 s; recortes e merge < 0,01 s. O PP-OCR de reconhecimento nem roda (só é fallback do TrOCR). A
    correção dos 13,2 s do p90 do docente é **trocar o TrOCR como leitor** pelo CRNN (3,3 ms por recorte em CPU).

## Limitações honestas

1. **Rótulo humano parcial** (C2/C3): 3.063 fotos de um lote e um rotulador, sem leitura e sem dupla; sem kappa humano × humano. O kappa humano × modelo (≈ 0,32) mostra desacordo de critério nos dois sentidos, com o modelo mais leniente em média. Itens D/E/L seguem com o proxy fraco de recortes.
2. **Detector e classificador de visor** (auxiliares ao bloco) foram treinados com split aleatório por foto; não alteram o bloco
   do lab, mas sua avaliação não segue a regra por lote.
3. **Limiar de detector calibrado em valid+test juntos** (vazamento leve, documentado em `docs/2026-10-08_cena_det_e_retest_amarelo.md`).
4. n pequeno (280 e 300 recortes de teste): diferenças < 3 pontos estão dentro do ruído (McNemar pareado em `leiturista compare-crnn`).
