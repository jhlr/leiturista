# Plano de treino — CRNNDigitos (Lab 2, bloco Luminus)

**Data:** 2026-09-27 · **Status:** plano, nada implementado ainda · **Serve de base para:**
`lab-02/ARQUITETURA.md` (seções 3, 11, 14) e `lab-02/USO-DE-IA.md`.

Substitui `docs/dl-lab2/plano-augmentations.md` (conteúdo incorporado abaixo, arquivo antigo
removido — um dono só pro plano, não dois documentos fragmentados).

## Como usar este documento

Este arquivo é a **especificação única** do bloco profundo do Luminus — não um resumo de
conversa, um documento autocontido o bastante pra alguém (ou um agente) implementar direto a
partir daqui, sem precisar reconstruir contexto de outro lugar. Cada seção diz **o quê**, **por
quê** e, quando aplicável, **onde no código** (arquivo/função). Decisão tomada = marcada
"decidido"; decisão em aberto = marcada "em aberto" com o critério de quando resolver.

### Rastreabilidade — onde cada item do enunciado é coberto

| Item do enunciado | Peso | Coberto em | Status |
|---|---|---|---|
| A — Ficha de arquitetura | 0,20 | Este doc alimenta `ARQUITETURA.md` (seções 2, 3, 11, 14) | Conteúdo pronto, falta transcrever pro `ARQUITETURA.md` |
| B — Modelo profundo em código (B1-B3) | 0,15 | `notebooks/lab02-dl.ipynb`, seções B1/B2/B3 | **Feito** |
| C — Rótulos e partição (C1-C4) | 0,20 | Seção "C — Rótulos e partição" abaixo (novo) | Esquema definido, rotulagem manual pendente |
| D — Sanidade e baseline (D1-D3) | 0,15 | D1/D2 em `lab02-dl.ipynb` (feito); D3 na seção "D3" abaixo (novo) | D1/D2 feitos, D3 especificado, não implementado |
| E — Confiança e recusa (E1-E2) | 0,10 | Seção "E — Confiança e recusa" abaixo (novo) | Especificado, não implementado |
| L1-L4 — específico Luminus | 0,20 (0,05 cada) | Seção "Luminus — L1 a L4" abaixo (novo) | Especificado, não implementado |
| Execução (roda do zero) | 20% da nota | `notebooks/lab02-dl.ipynb` já executa do zero | Válido pro que já existe |

## Revisão externa (agente sem contexto, persona professor MLOps/DL/CV, 2026-09-27)

Pedido pra revisar o plano do jeito que um professor rigoroso revisaria antes do **primeiro
status report** (não o entregável final — prazo 08/10, ~11 dias). Veredito: direção certa,
disciplina de avaliação acima da média (checkpoint por valid real, teste olhado uma vez, seed
única), mas 5 correções necessárias antes do status report — incorporadas abaixo, cada uma no
lugar do plano que ela corrige:

1. **Vazamento de informação entre config 1 e config 2** (não vazamento de teste clássico, mas
   real): o plano olhava o teste da config 1 (estágio 1 sozinho) e só depois decidia
   hiperparâmetro do estágio 2 — a decisão já nasce informada pelo número de teste da config
   anterior. Correção: hiperparâmetros do estágio 2 (LR, clipping, épocas) se fixam **antes** de
   olhar o teste da config 1, não depois. Ver Estágio 1/2 abaixo.
2. **300 exemplos de teste é pouco pra decidir entre 3 configs sem teste estatístico** — erro
   padrão de proporção ~2,3pp nessa faixa, uma diferença de "0,880 pra 0,90" pode ser ruído.
   Correção: aplicar **McNemar pareado** (mesmo conjunto de teste nas 3 configs) além do número
   bruto de leitura exata. Ver seção "Comparação estatística" abaixo.
3. **Nenhum orçamento de tempo pro estágio 1 em CPU** — risco real pro cronograma de 11 dias se
   o pré-treino sintético sozinho consumir dias. Correção: medir 1 época numa fração pequena
   (ex. 500 imagens sintéticas) e extrapolar **antes** de comprometer com o fluxo completo.
4. **Critério de calibração do clipping era "olhar a norma", não reproduzível.** Correção:
   percentil 90 da norma de gradiente observada nas primeiras N iterações sem clipping vira o
   valor de corte — não "olhar e decidir".
5. **CTC em sequência sintética de comprimento 1-2 é caso degenerado conhecido** (sem repetição
   a colapsar, alinhamento trivial) e o plano amostrava esse comprimento da distribuição
   empírica (que já é rara, <2%) sem declarar o risco. Fica registrado como risco conhecido —
   não bloqueia o status report, mas se o treino do estágio 1 for instável, é o primeiro
   suspeito a checar.

O revisor também confirmou como "pode ficar em aberto por enquanto" (não vale gastar tempo
agora): valor exato do clip, forma exata do scheduler (cosine vs. linear — decide com a curva
de loss do estágio 1), o eval dedicado de glare/oclusão, o Estágio 3 (depende de C/dado
externo), e o tamanho de batch (sai da primeira rodada de sanidade).

## Objetivo

Definir o fluxo de treino inteiro do bloco profundo do Luminus (`CRNNDigitos`, o reconhecedor
de dígitos do display recortado — parte B do enunciado) em estágios, com a justificativa e a
alternativa descartada de cada decisão, do jeito que a seção A do enunciado cobra.

## Por que estágios, e não treinar direto no UFPR-AMR

Já treinamos direto: `notebooks/lab02-dl.ipynb`, `CRNNDigitos` do zero em
`data/finetune_ufpramr` (1.400 recortes reais), 15 épocas, sem augmentation nova — deu
**leitura exata 0,880 / acurácia por dígito 0,948 no teste**. Isso já é uma baseline real, não
um placeholder. A pergunta que este plano responde não é "dá pra treinar" (já sabemos que dá),
é: **pré-treinar num dataset maior de dígito de display antes do UFPR-AMR melhora a
generalização, ou só gasta tempo?** — pergunta que só se responde comparando os dois
caminhos com o mesmo protocolo de avaliação, não assumindo que pré-treino ajuda.

## Os dois datasets em jogo

| Dataset | O que é | Tamanho | Rótulo | Domínio |
|---|---|---|---|---|
| `data/finetune_ufpramr` | Recortes de display já extraídos do UFPR-AMR (Laroca 2020) | 1.400 treino / 300 valid / 300 teste | sequência de dígitos (1-5 dígitos, real) | Medidor de energia elétrica, o domínio-alvo do projeto |
| `markuspfeifer/lcd-digits` (Kaggle, CC0) | Fotos de multímetro digital `emuig M3900` | **34.033 imagens, um dígito por imagem**, já separadas em pastas `0/`...`9/` (3.896 a 5.987 por classe) | classe de dígito único (0-9), **não é sequência** | Display LCD genérico (multímetro, não medidor de energia) |

**Achado que muda o plano**: o LCD digits não é um dataset de leituras multi-dígito com
transcrição — é um dataset de **classificação de dígito isolado**. Não dá pra jogar direto no
`CTCLoss` do `CRNNDigitos` (que espera uma sequência por imagem). Duas formas de usar,
descritas abaixo — a segunda é a recomendada.

### Opção A (descartada como principal) — pré-treinar só a CNN como classificador de dígito único

Treinar `r.cnn` (a parte convolucional de `CRNNDigitos`) com uma cabeça `Linear(128, 10)` +
`CrossEntropyLoss` sobre os dígitos isolados, depois transplantar os pesos da CNN pro
`CRNNDigitos` completo antes do fine-tune com CTC em `finetune_ufpramr`.

- **Por que é opção, não a principal:** só pré-treina a CNN (extrator visual), a GRU nunca vê
  nada além de ruído aleatório até o fine-tune — perde exatamente a parte que aprende a "ler da
  esquerda pra direita" e a colapsar repetição via CTC, que é o componente mais específico do
  reconhecedor de sequência.

### Opção B (recomendada) — compor sequências sintéticas e pré-treinar o `CRNNDigitos` inteiro

Concatenar horizontalmente `N` dígitos isolados (amostrados aleatoriamente das pastas `0-9`)
pra montar uma imagem de "leitura sintética" com rótulo sequencial conhecido, e treinar o
`CRNNDigitos` completo (CNN + GRU + CTC) nessas sequências sintéticas antes do fine-tune real.

- **`N` (comprimento) amostrado da distribuição empírica real do UFPR-AMR**, não uniforme:
  medido em `finetune_ufpramr/labels.csv` — 54,9% dos rótulos têm 5 dígitos, 39,6% têm 4,
  3,75% têm 3, e os 1-2 dígitos são raros (<2% juntos). Sortear `N` dessa distribuição em vez
  de uniforme(1,5) evita que o pré-treino ensine o modelo a esperar sequências curtas que quase
  não existem no domínio real.
- **Por que é a opção principal**: treina o `CRNNDigitos` inteiro (CNN+GRU+CTC) na mesma tarefa
  estrutural do fine-tune (ler sequência via CTC), só que com textura de dígito diferente —
  exatamente o que "pré-treino antes do fine-tune" deveria significar aqui.
- **Ressalva honesta pro `ARQUITETURA.md`**: dígitos justapostos artificialmente não têm o
  espaçamento, o brilho de fundo contínuo nem o estilo de fonte consistente de um display real
  de 4-5 dígitos — é dado sintético "melhor que ruído", não dado real a mais. Precisa comparar
  com e sem esse pré-treino, não assumir que ajuda (mesma lógica do K2 do Korvian: rótulo fraco
  sintético avaliado contra dado real, não aceito de graça).

## Fluxo de treino proposto (3 estágios)

### Estágio 0 — sanidade (já feito, `notebooks/lab02-dl.ipynb`)

D1 (perda inicial vs. referência ln(11)) e D2 (sobreajuste de lote pequeno, hoje com dado
sintético trivial) já passam. Não muda.

### Estágio 1 — pré-treino em sequências sintéticas do LCD digits

0. **Medir orçamento de tempo antes de comprometer com o fluxo inteiro** (correção do revisor,
   item 3): rodar 1 época numa fração pequena (ex. 500 sequências sintéticas) em CPU, extrapolar
   pro dataset cheio (34k dígitos → sequências) × nº de épocas planejado. Se a extrapolação
   estourar a folga dos 11 dias até o status report, cortar aqui (menos épocas, menos dado
   sintético) antes de rodar o resto, não depois.
1. ~~Baixar `markuspfeifer/lcd-digits`~~ — feito, ver "Próximos passos".
2. Montar `N` dígitos por amostra, comprimento sorteado da distribuição empírica do UFPR-AMR
   (acima), imagens redimensionadas e concatenadas em escala de cinza. **Risco conhecido, não
   bloqueante** (correção do revisor, item 5): comprimento 1-2 é caso degenerado pra CTC (sem
   repetição a colapsar, alinhamento quase trivial) e já é raro na distribuição real (<2%) — se
   o treino do estágio 1 ficar instável, checar esses casos primeiro antes de qualquer outra
   hipótese.
3. Aplicar o conjunto **cheio** de augmentation (lista abaixo) — dado abundante (34k), pode ser
   agressivo sem medo de esgotar variedade.
4. Treinar `CRNNDigitos` do zero nessas sequências sintéticas, com gradient clipping (critério
   de calibração: percentil 90 da norma de gradiente observada nas primeiras N iterações sem
   clipping, não "olhar e decidir" — correção do revisor, item 4) e scheduler de LR (warmup +
   decay — ver "Lacunas", itens 3-4).
5. **Fixar TODOS os hiperparâmetros do estágio 2 (LR, clipping, épocas) antes de rodar o passo
   6** (correção do revisor, item 1 — vazamento de informação entre configs). Documentar a
   decisão por escrito (neste arquivo ou no `README.md` do `lab-02/`) antes de olhar qualquer
   número de teste.
6. **Avaliar contra `valid` E `test` REAIS do `finetune_ufpramr` já neste estágio** — decisão do
   usuário, e resolve de uma vez a lacuna 1 da seção "Lacunas" (validar só em holdout sintético
   cria ponto cego pro que importa, que é o dado real). Seleção de checkpoint (lacuna 5): o
   melhor estado por época é escolhido pela leitura exata no `valid` real, não no sintético e
   não pela última época. `test` real roda **uma vez** ao final do estágio 1 (checkpoint já
   escolhido) — essa é a "config 1" da comparação; olhar de novo só depois do estágio 2 (config
   2), não a cada mudança. Os hiperparâmetros do estágio 2 já estão fixos desde o passo 5, então
   olhar esse número agora não influencia mais a config 2.

### Estágio 2 — fine-tune em `finetune_ufpramr` (dado real do domínio)

1. Carregar os pesos do estágio 1 (em vez de `CRNNDigitos()` do zero, como no
   `notebooks/lab02-dl.ipynb` atual).
2. Aplicar a augmentation "núcleo" (lista abaixo) — dado real e menor (1.400), menos agressivo
   que o estágio 1 pra não distorcer demais uma amostra já pequena.
3. **Sem congelamento — decisão do usuário.** A ideia original (congelar a CNN nas primeiras
   épocas) fazia sentido pra transfer learning de um backbone genérico (tipo ImageNet), onde o
   extrator vem de uma tarefa bem diferente. Aqui não é o caso: o estágio 1 já treina o
   `CRNNDigitos` inteiro (CNN+GRU+CTC) do zero, na mesma tarefa estrutural (ler sequência via
   CTC) que o estágio 2 — não há um extrator "genérico" a proteger de esquecimento catastrófico.
   Fine-tune roda com tudo treinável desde a época 0. B2 do notebook (`lab02-dl.ipynb`) continua
   reportando a contagem de parâmetros nos dois regimes porque é isso que o enunciado pede
   (extração × ajuste fino, item B2) — mas isso é só a contagem, não implica rodar o treino
   congelado de verdade.
4. Gradient clipping + scheduler de LR (mesma lógica do estágio 1, ajustado pro dataset menor).
   Seleção de checkpoint por `valid` real, igual ao estágio 1.
5. Comparar contra a baseline já medida (treino do zero, sem estágio 1): leitura exata 0,880 /
   acurácia por dígito 0,948 no teste. Reportar os três números lado a lado — baseline (sem
   pré-treino), estágio 1 sozinho (zero-shot no domínio real), estágio 1 + estágio 2 — se o
   pré-treino não ajudar ou piorar, isso é resultado válido pro `README.md`, não motivo pra
   esconder. **Ver "Comparação estatística" abaixo antes de declarar qualquer config vencedora.**

### Comparação estatística entre as 3 configs (correção do revisor, item 2)

300 exemplos de teste real é pouco pra decidir entre 3 configs só olhando o número bruto de
leitura exata — erro padrão de proporção ~2,3 pontos percentuais nessa faixa, então uma
diferença de "0,880 pra 0,90" pode ser ruído de amostragem, não sinal real. Como as 3 configs
(baseline, estágio 1 sozinho, estágio 1+2) rodam **no mesmo conjunto de teste** (300 recortes
fixos de `finetune_ufpramr`), o teste certo é **McNemar pareado** (compara acerto/erro por
exemplo entre duas configs, não a proporção agregada) — não um teste de proporções independentes.
Rodar McNemar nos 3 pares (baseline×estágio1, baseline×estágio1+2, estágio1×estágio1+2) antes de
escrever no `README.md` qual configuração "venceu". Se nenhum par for estatisticamente
significativo, isso também é resultado válido a reportar — "as três configs empatam dentro do
ruído amostral" é uma conclusão honesta, diferente de "o pré-treino não ajudou".

### Estágio 3 — futuro, depende do item C (rotulagem) ou de `data/distribuidora_amr`

Fine-tune/avaliação final em recorte real da distribuidora. `docs/fotos_distribuidora_dados.md`
(2026-09-14) registra que esse dataset já foi construído uma vez (1.087 recortes aceitos, split
869/108/110) numa sessão Windows — **não existe neste Mac** (`data/distribuidora_amr` não está
no disco aqui). Decidir com o usuário se replica o `leiturista import-distribuidora` aqui ou
usa como está quando disponível. É o que L2 do enunciado pede (benchmark em ≥200 recortes reais
da distribuidora).

## Augmentations por estágio

### Núcleo — usar nos dois estágios (1 e 2), sem mudar parâmetro

| Augmentation | Parâmetro | Por quê |
|---|---|---|
| Brilho/contraste | `alfa∈[0.7,1.3]`, `beta∈[-30,30]` | Já escrita em `02_tratamento.ipynb`; iluminação de campo varia |
| Giro pequeno | `±10°` | `02_tratamento.ipynb` já mostrou que corrigir rotação geometricamente é frágil (Hough); vira augmentation |
| Desfoque leve | Gaussian 5×5, prob. 0,3 | Foco de câmera de PDA |

### Específicas do estágio 2 (fine-tune, dado real de domínio) — prioridade alta

| Augmentation | Parâmetro | Por quê |
|---|---|---|
| **Inversão de polaridade** (branco↔preto) | ~40% dos exemplos | 41,7% dos 1.087 recortes reais da distribuidora vieram da fase invertida (`fotos_distribuidora_dados.md`) — não é exceção, é quase metade dos casos reais |
| **Jitter de recorte** (crop/zoom ±5-10%) | padding/corte aleatório de borda | O det (PP-OCRv5) não entrega caixa pixel-perfeita |

### Específicas do estágio 1 (pré-treino sintético, dado abundante) — prioridade média/alta

O estágio 1 é o único com uma etapa de **montagem** (concatenar `N` dígitos isolados numa
sequência) — isso cria um ponto de decisão que não existe no estágio 2 (lá o recorte já vem
inteiro): **augmentation antes ou depois de concatenar?**

**Decisão: glare e oclusão parcial entram ANTES de concatenar, por dígito individual — não
depois, na sequência montada.** Motivo: reflexo e obstrução no dado real de campo não cobrem a
tira inteira, cobrem 1-2 dígitos (uma sombra de dedo, um brilho localizado no vidro sobre uma
parte do display). Uma elipse de glare desenhada por cima da sequência já montada tende a cortar
ao meio de vários dígitos ao mesmo tempo com a mesma máscara — não é o padrão real. Aplicando por
dígito, antes da montagem, cada posição da sequência recebe (ou não) degradação
independentemente, o que aproxima mais do padrão real de "um dígito comprometido, os outros
legíveis" — e é exatamente o caso que motivou a pergunta (reflexo/obstrução parcial).

| Augmentation | Quando aplicar | Parâmetro | Por quê |
|---|---|---|---|
| **Glare/reflexo sintético** | **pré-concatenação, por dígito** | elipse clara semi-transparente, opacidade 0,3-0,6, aplicado independentemente em ~15-20% dos dígitos (não da sequência inteira) | Primeiro item da taxonomia `DEFEITOS` do esqueleto do professor; por-dígito reproduz o padrão real de degradação localizada, não uniforme na tira |
| **Oclusão parcial (cutout)** | **pré-concatenação, por dígito** | retângulo cobrindo até ~30% da **área do dígito individual**, nunca o dígito inteiro, em ~10-15% dos dígitos | Mesma lógica do glare — sujeira/obstrução real atinge parte de 1-2 dígitos, não a tira toda. Aplicar por dígito ANTES de montar também resolve o risco de "cobrir um dígito inteiro sem querer" que esse augmentation tinha na v1 do plano: aqui o cutout é relativo ao recorte de um único dígito, então dá pra limitar a fração coberta com garantia, em vez de sortear um retângulo solto sobre a sequência inteira e torcer pra não apagar nada |
| Ruído leve (gaussiano/sal-e-pimenta) | pré-concatenação, por dígito | σ baixo, ~3-5% dos pixels | Ruído de sensor — também plausivelmente desigual entre dígitos (parte do display com reflexo tem ruído diferente da parte limpa) |
| Jitter de aspect ratio no resize | **pós-concatenação**, na sequência montada | ±15% na largura de destino | Afeta a tira inteira igualmente (é o resize final); 32×128 é placeholder do esqueleto (ver B1 em `lab02-dl.ipynb`), não medida do projeto |

### Opcional / baixa prioridade

Compressão JPEG sintética (qualidade 40-70), aplicada pós-concatenação (é um artefato da
imagem final salva pelo app do PDA, não de um dígito isolado).

### Descartadas — e por quê

| Augmentation | Motivo |
|---|---|
| Espelhamento horizontal/vertical | Dígito espelhado não existe fisicamente — rótulo impossível (decidido em `02_tratamento.ipynb`) |
| Rotação grande (>15°) | Pipeline já corrige rotação grosseira antes do recorte; girar mais foge da distribuição real e estica demais o eixo de tempo que a CTC usa |
| Deformação elástica | Distorce o segmento de 7-segmentos de um jeito que não acontece fisicamente; gasta capacidade do modelo tolerando degradação inexistente |
| Blend de dígito em transição (contador mecânico) | Real pra medidor ciclométrico, mas os dois datasets em jogo (UFPR-AMR e LCD digits) são só display digital, sem transição — fica só anotado como frente futura |
| Hue jitter | Pipeline já converte pra escala de cinza antes do recognizer; não se propaga |

## Lacunas encontradas via pesquisa (a incorporar no fluxo)

Pesquisa rápida sobre boas práticas de treino de CRNN+CTC, pré-treino sintético→fine-tune real,
e checklist de reprodutibilidade em ML trouxe pontos concretos que faltavam neste plano:

1. **Validação do estágio 1 não pode ser só sintética — decidido.** Achado direto da busca
   (domain-gap em pré-treino sintético): "hold out a validation set drawn from real examples,
   not synthetic ones — synthetic validation data creates a blind spot for failure modes".
   **Decisão do usuário**: `valid` **e** `test` reais de `finetune_ufpramr` entram desde o
   estágio 1, não só depois do fine-tune (ver Estágio 1, item 5, acima).
2. **Sem congelamento — decidido.** O usuário apontou o ponto certo: congelar só faz sentido
   protegendo um extrator vindo de uma tarefa/domínio bem diferente (tipo ImageNet); aqui o
   estágio 1 já treina o `CRNNDigitos` inteiro do zero na mesma tarefa (CTC), então não existe
   "extrator genérico" a proteger. O achado da busca (queda de até 13,5% ao congelar, num caso de
   vídeo/transfer learning) reforça a lógica, mas a decisão veio do reconhecimento de que a
   premissa do congelamento não se aplica aqui, não da citação. Fine-tune roda com tudo treinável
   desde a época 0 (Estágio 2, item 3).
3. **Gradient clipping — decidido, entra nos dois estágios.** Norm global (`torch.nn.utils.
   clip_grad_norm_`). **Critério de calibração (correção do revisor, item 4, não é mais "olhar e
   decidir")**: rodar as primeiras N iterações sem clipping, medir a norma de gradiente
   observada, fixar o corte no **percentil 90** dessa distribuição — reproduzível, não
   arbitrário. CTC é conhecido por instabilidade de gradiente em sequências longas/alinhamentos
   degenerados; `zero_infinity=True` já protege contra `nan`/`inf` explícito no `CTCLoss` atual,
   clipping cobre a instabilidade que não chega a `inf`.
4. **Scheduler de LR (warmup + decay) — decidido, entra nos dois estágios.** Warmup curto
   (poucas centenas de passos ou primeiras 1-2 épocas) + decay depois (cosine ou linear).
   Aplicado nos dois estágios por consistência, mesmo o estágio 1 (dataset grande) tolerando LR
   fixo melhor que o estágio 2 (dataset pequeno).
5. **Seleção de checkpoint por valid — decidido.** Guardar o melhor estado por leitura exata no
   `valid` **real** (não sintético — consistente com o item 1) a cada época, nos dois estágios;
   esse checkpoint, não o da última época, é o avaliado em teste.
6. **Disciplina de olhar o teste uma vez por configuração.** Já é regra geral do projeto e do
   enunciado (penalização explícita por "limiar/temperatura ajustados no teste"), fica explícito
   aqui: hiperparâmetro/checkpoint se escolhe com `valid`; `test` roda uma vez por configuração —
   agora são **três** configurações a comparar (baseline sem pré-treino, estágio 1 sozinho,
   estágio 1+2), não duas — e o número fica registrado, não se reavalia tentando bater melhor.
7. **Seed única para as duas bibliotecas — decidido.** `torch.manual_seed(SEED)` e
   `np.random.default_rng(SEED)` passam a usar o **mesmo** valor (`SEED = 42`), em vez do 42/0
   divergente de antes. Um valor só documentado por rodada, cobrindo modelo/otimizador (torch) e
   embaralhamento/sorteio de sequência sintética (numpy) igualmente.

Fontes da pesquisa: [Deep Learning Tuning Playbook — Google](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq),
[Stabilization Techniques (clipping/LR/warmup)](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-24-identifying-mitigating-training-instabilities/stabilization-techniques-revisited),
[The Warmup Dilemma — LR strategies e convergência em speech-to-text](https://arxiv.org/pdf/2505.23420),
[Training Deep Networks with Synthetic Data — domain randomization](https://arxiv.org/pdf/1804.06516),
[An Evaluation of Large Pre-Trained Models for Gesture Recognition using Synthetic Videos (achado do congelamento piorando)](https://arxiv.org/pdf/2410.02152),
[The Machine Learning Reproducibility Checklist v2.0 (McGill)](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf).

## O que fica pra depois de rodar (não assumir agora)

- Se o pré-treino sintético (estágio 1) realmente melhora valid/test do estágio 2, ou é neutro,
  ou piora — reportar o que sair, é exatamente o tipo de comparação que D3/L3 do enunciado pedem.
- Quanto tempo de treino o estágio 1 custa em CPU com 34k imagens sintéticas — não medido ainda,
  decide o orçamento de época antes de rodar full.
- **A hipótese "augmentation pré-concatenação ajuda em reflexo/obstrução" precisa de um eval
  dedicado pra ser verificada, não só assumida.** `finetune_ufpramr`/`test` não tem rótulo de
  "esse recorte tinha reflexo" — leitura exata/dígito agregada não isola o efeito. Preparar um
  pequeno conjunto de teste sintético (sequências com glare/oclusão forçados, reserva separada
  do treino) pra comparar o modelo com e sem essa augmentation especificamente nesses casos,
  além da métrica geral em valid/test.

## C — Rótulos e partição (0,20 — a maior fatia, ainda não iniciada)

Dado bruto já em mãos: `data/distribuidora_campo/` — 4 lotes reais (`PSP_EXTRATLEITIMPL_030726_0121`,
`_200526_0352`, `_210526_0335`, `_220526_0408`), cada um com CSV (`Numero do medidor;Posicao do
medidor lida;Nota de Leitura Atual;Foto do medidor`) + fotos do medidor inteiro. Existe também
`data/distribuidora_campo_rotulado.csv` (13.695 linhas) com colunas `llm_*` — **rotulagem por
LLM, não humana**, não serve pra C2/C3 (kappa exige dois humanos às cegas), mas serve como
**triagem**: usar pra escolher as 300 fotos de C2 estratificadas por classe estimada, em vez de
sortear cego e arriscar desbalancear legibilidade.

- **C1 — esquema (`lab-02/rotulos/esquema.md`)**: duas dimensões de rótulo por foto, não uma.
  (a) **Legibilidade** — categórica, é o que entra no kappa: `legível` / `parcialmente_legível`
  / `ilegível` / `não_é_medidor`. Dois exemplos-limite por classe (a `llm_confidence` baixa do
  CSV de triagem já aponta candidatos pra fronteira `legível`/`parcialmente_legível`).
  (b) **Leitura transcrita** — string de dígitos, só quando `legível` ou `parcialmente_legível`;
  concordância entre rotuladores medida por exact-match/distância de edição, não kappa (kappa é
  pra categórica).
- **C2 — 300 fotos**, estratificadas pelos 4 lotes (proporcional ao tamanho de cada lote, não
  igual) e pela classe estimada via `llm_quality`/`llm_display_readable` do CSV de triagem (pra
  não sair só com fotos fáceis). Com 5-7 integrantes: 45-60 fotos/pessoa.
- **C3 — 50 com dupla rotulagem às cegas**: planilhas separadas por rotulador até os dois
  terminarem (regra do enunciado). Reportar kappa de Cohen por classe de legibilidade; kappa <
  0,6 numa classe = esquema ambíguo, reescrever e registrar o que mudou.
- **C4 — partição por lote**: os 4 lotes acima já são a unidade natural de partição (não
  aleatória por foto) — nenhum medidor/leitura pode aparecer em dois splits. Como só há 4 lotes
  pra 3 splits (treino/valid/teste), decidir explicitamente o mapeamento lote→split (ex.: 2
  lotes treino, 1 valid, 1 teste) e declarar por que essa divisão, não outra.
- **Ligação direta com L2**: os recortes de display das 300 fotos de C2 (ou subconjunto ≥200)
  viram o benchmark real que L2 pede — não é trabalho duplicado, C2 alimenta L2.

## D3 — baseline não profunda (falta, D1/D2 já feitos em `lab02-dl.ipynb`)

Comparar `CRNNDigitos` contra um classificador **não profundo** no mesmo `valid`/protocolo.
Proposta concreta: já que os recortes têm proporção largura/altura consistente (~2:1 a 5:1) e
comprimento de sequência conhecido por amostra, segmentar cada recorte em `N` fatias de largura
igual (uma por dígito esperado), extrair feature clássica por fatia (HOG, ou PCA sobre pixel
bruto redimensionado), e treinar regressão logística multinomial 10-classes por posição — sem
rede neural, sem CTC. Rodar com **3 seeds** (shuffling/init), reportar média±desvio de leitura
exata no mesmo `valid` real usado pelo `CRNNDigitos`. Se a baseline perder (esperado, já que
ignora a estrutura de sequência e depende de segmentação por largura fixa, que quebra em
recortes com espaçamento irregular), isso vale nota cheia segundo o enunciado — desde que
digam por quê.

## E — Confiança e recusa (falta, 0,10)

- **E1 — ECE + reliability diagram**: usar a confiança por sequência (produto ou mínimo das
  probabilidades softmax por passo de tempo decodificado, via `logits.softmax(-1)` de
  `CRNNDigitos.forward`) como score de confiança. Calcular ECE no `valid` real antes e depois de
  ajustar um único parâmetro de temperatura (`logits / T`, `T` otimizado no `valid`, nunca no
  teste) por log-verossimilhança negativa.
- **E2 — curva cobertura×risco**: usar o mesmo score de confiança como critério de recusa —
  variar o limiar, plotar % de sequências aceitas (cobertura) vs. taxa de erro nas aceitas
  (risco). Marcar no gráfico a meta de negócio declarada no kickoff — **checar
  `docs/pedido_kickoff.md`/`docs/imersao_entendimento_objetivos_completo.md` pelo número real
  antes de inventar um "cobertura ≥ X, risco ≤ Y"**; se não houver meta declarada com número,
  dizer isso explicitamente em vez de fabricar um alvo (regra de honestidade epistêmica do
  projeto).

## Luminus — L1 a L4 (0,20, 0,05 cada — específico do grupo)

- **L1 — CRNN+CTC vs. TrOCR, 10 erros reais de cada.** Já dá pra fazer **sem esperar C**: rodar
  os dois modelos no mesmo `test` real de `finetune_ufpramr` (300 recortes) — `CRNNDigitos` já
  treinado (0,880 leitura exata) e a baseline TrOCR/PP-OCR já medida no artigo SBTI
  (`docs/sbti_artigo/`, 0,357/0,846). Precisa: (a) modificar `avaliar()` em `lab02-dl.py` pra
  também salvar a predição por exemplo (hoje só agrega a métrica), (b) rodar o pipeline TrOCR
  existente (`src/leiturista/inference.py`) nos mesmos 300 recortes de teste do UFPR-AMR, (c)
  pegar 10 erros de cada modelo e classificar manualmente em perdeu/duplicou/trocou/inventou
  dígito.
- **L2 — benchmark ≥200 recortes reais da distribuidora com transcrição.** Depende de C2 (ver
  acima — a ligação já está desenhada, não é trabalho extra).
- **L3 — declarar a métrica otimizada + 2 restrições numéricas.** Proposta: otimizar **leitura
  exata** (não acurácia por dígito) — justificativa de custo: um único dígito errado numa
  leitura de consumo já torna a fatura errada (não existe "quase certo" pra faturamento), então
  acurácia por dígito mascara o que realmente importa. Restrições candidatas (a confirmar com
  número real, não específico ainda): acurácia por dígito ≥ 0,90 como piso de qualidade, e
  latência p90 ≤ valor já medido no artigo (13,2s) como teto — mas isso é do pipeline PP-OCR
  atual, não do `CRNNDigitos` novo, então precisa medir a latência do `CRNNDigitos` em CPU
  separadamente antes de declarar o mesmo teto.
- **L4 — plano de ajuste fino + onde estão os 13,2s do p90, por estágio.** O "plano de ajuste
  fino" já está especificado nos Estágios 1-2 acima (congelamento, taxas, augmentation
  permitida/proibida). O que falta e não está feito: **instrumentar `src/leiturista/
  inference.py` pra medir tempo por estágio** (det PP-OCRv5, correção de rotação, rec) em vez de
  só o tempo agregado do pipeline — hoje o artigo só tem o número fim-a-fim (13,2s p90, 382
  imagens), não sabe se o gargalo é detecção, rotação ou reconhecimento.

## Cronograma (dado o prazo 08/10, ~11 dias de hoje 2026-09-27)

Ordem de prioridade sugerida, considerando peso na nota e dependência entre itens:

1. **C1/C2/C3 (rotulagem)** — maior peso isolado (0,20) e bloqueia L2; começar primeiro, dica
   nº1 do próprio enunciado ("comecem pela rotulagem").
2. **Estágio 1 + Estágio 2 (treino)**, em paralelo à rotulagem por outra parte do grupo — já
   especificado, só falta implementar.
3. **L1** (não depende de C, pode rodar assim que o treino do estágio 2 terminar).
4. **D3, E1/E2** — depois do treino principal, reusam os mesmos checkpoints/valid.
5. **L2, L3, L4** — depois de C2 pronto e do treino terminado.
6. **ARQUITETURA.md, README.md, USO-DE-IA.md** — consolidação final, últimos dias.

## Próximos passos

1. ~~Baixar `markuspfeifer/lcd-digits`~~ — **feito** (2026-09-27, via `kagglehub`, sem precisar de
   `kaggle.json`: dataset é público/CC0). 34.020 imagens em `data/lcd_digits/data/0`...`9`,
   gitignored.
2. Escrever o gerador de sequência sintética (Opção B) em `notebooks/lab02-dl.py`, com
   augmentation pré-concatenação (glare/cutout/ruído por dígito) e pós-concatenação
   (aspect-ratio jitter, JPEG).
3. Implementar gradient clipping, seleção de checkpoint por valid, e (no mínimo pro estágio 2)
   warmup+decay de LR — itens 3-5 da seção de lacunas acima.
4. Rodar estágio 1 (pré-treino), avaliar contra `valid` real de `finetune_ufpramr` mesmo antes do
   fine-tune (item 1 da seção de lacunas), depois estágio 2 (fine-tune) carregando os pesos do
   estágio 1, comparar contra a baseline já medida (0,880/0,948, treino do zero sem pré-treino).
5. Registrar o resultado da comparação em `lab-02/README.md`; registrar as escolhas
   aceitas/descartadas deste documento em `lab-02/USO-DE-IA.md`.
6. Abrir C1/C2/C3 (esquema + rotulagem manual, ver seção "C" acima) — em paralelo aos passos
   1-5, não depois; é o item de maior peso isolado e não depende do treino terminar.
7. Depois do estágio 2 rodar: L1 (comparação de erros, não depende de C), D3 (baseline não
   profunda), E1/E2 (calibração e cobertura×risco).
8. Depois de C2 pronto: L2 (benchmark real), C4 (partição por lote formalizada), L3/L4
   (métrica declarada, latência por estágio).
9. Consolidar tudo em `lab-02/ARQUITETURA.md`, `lab-02/README.md`, `lab-02/USO-DE-IA.md` usando
   a tabela de rastreabilidade no topo deste documento como checklist.

## Implementação e achados (2026-10-08)

`src/leiturista/synth.py` (gerador: comprimento da distribuição real, glare/cutout/ruído por
dígito, crop central do dígito, jitter de largura 85-100%) + `train_crnn` estendido
(`--synth-per-epoch`, `--synth-only`, `--warmup`, `--cosine`, `--clip-calib-iters`,
`--crop-jitter`, `--rot-deg`, `--blur-p`, `--aug-start`, `--norm`, `--width`; checkpoint escolhido
pelo valid real; fine-tune herda norm/width do checkpoint).

Achados (todos medidos em `data/finetune_ufpramr`, test = 300 recortes):

1. **GroupNorm não serve aqui.** Sobreajuste de 1 lote fixo de 32 recortes, 150 passos: BatchNorm
   128 = 1,00 de leitura exata; GroupNorm 128 ou 160 = 0,00. Padrão voltou a `batch`. Largura 160
   também atrasa (0,72), padrão continua 128.
2. **Augmentation alonga o platô inicial da CTC** (modelo só emite brancos, perda ~2,65). Real-only
   com augment completo, 8 épocas: travado. Só brilho/contraste: escapa. Inversão (p=0,4) sozinha
   ou jitter de recorte sozinho: ainda travados em 8 épocas. Giro e desfoque não pesaram.
   Sintéticos não atrapalham (mistura sem augment: 0,583 em 8 épocas).
3. **Correção na causa: currículo `--aug-start 6`** (6 primeiras épocas sem augmentation, depois
   tudo ligado). 500 sintéticos + 1.400 reais, augment em tudo, BatchNorm 128:
   15 épocas = 0,817 / 0,927; **30 épocas = 0,873 / 0,945** (exata / por dígito, test).
   Baseline do notebook (só real, sem augment, 15 épocas): 0,880 / 0,948. Diferença de 0,007 em
   300 exemplos = empate dentro do ruído (falta McNemar).
4. Os crops do LCD digits são sujos (dígito cortado, vizinhança); o crop central ajudou a limpar.

```bash
# melhor config até aqui (mistura + currículo)
caffeinate -i .venv/bin/leiturista train-crnn --synth-per-epoch 500 --epochs 30 --warmup 20 --clip-calib-iters 20 --invert-prob 0.4 --crop-jitter 0.07 --aug-start 6 --out models/crnn_bn128_mix500_cur30.pt --experiment crnn-mix --tracking-uri sqlite:///data/mlflow_crnn.db
```
O `mlflow.db` principal (1,2G) está com schema defasado para o mlflow 3.16; esses runs foram para
`data/mlflow_crnn.db`. Falta: McNemar pareado entre as configs; estágio 2 puro (fine-tune a
partir de estágio 1 só sintético).

### Resultado das configs do plano (2026-10-08, test UFPR-AMR, n=300, McNemar exato pareado)

| Config | Leitura exata | Por dígito |
|---|---|---|
| Baseline (só real, sem augment, 15 ep) `crnn_digitos.pt` | 0,873 | 0,948 |
| Estágio 1 sozinho (só sintético, 15x4000 seqs) | 0,010 | 0,288 |
| Estágio 1 + estágio 2 (fine-tune real, lr 3e-4, 30 ep) | 0,857 | 0,951 |
| Mistura 500 sint. + real, currículo `--aug-start 6`, 30 ep | 0,873 | 0,945 |

McNemar: baseline x estágio 2 p=0,500; baseline x mistura p=1,000; estágio 2 x mistura p=0,442.
Estágio 1 x qualquer outra: p<0,001. **Conclusão honesta: baseline, estágio 1+2 e mistura empatam
dentro do ruído amostral; o pré-treino sintético não melhorou o UFPR-AMR.** O estágio 1 sozinho não
generaliza pro dado real (gap de domínio: LCD digits de 7 segmentos vs. displays do UFPR-AMR).
Comando: `leiturista compare-crnn <ckpts...>`.
