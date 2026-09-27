# Laboratório 2 — A arquitetura profunda do seu projeto

**Vale 1,0 ponto** · Entrega **por grupo do Projeto 4** · Prazo: **08/10/2026, 23h59**
Repositório do grupo, pasta `lab-02/`, link no Classroom.
Material de apoio: [`guia-arquitetura-rede-profunda.md`](guia-arquitetura-rede-profunda.md) e
[`codigo/esqueleto_arquitetura.py`](codigo/esqueleto_arquitetura.py).

---

## Por que este laboratório

Os quatro grupos atendem o mesmo cliente (distribuidora), com os mesmos 12.340 JPGs
e as mesmas quatro extrações diárias. No kickoff de 12/09 e nas apresentações de 17/09, cada
grupo mostrou um estágio diferente: há quem tenha plataforma inteira desenhada e nenhum
modelo, quem tenha OCR rodando e nenhuma calibração, quem tenha o objetivo fechado e nenhuma
arquitetura, e quem tenha a cascata desenhada e nenhum rótulo.

Este laboratório não pede o modelo final. Pede que o grupo **transforme o que já tem numa
arquitetura profunda justificada, verificada e com o primeiro tijolo treinado** — o que é
exatamente o que o Marco 2 (29/10) vai cobrar funcionando.

---

## Onde cada grupo está (leitura do docente sobre o material de 12/09 e 17/09)

| Grupo | O que já existe | O que falta para haver uma rede profunda |
|---|---|---|
| **Korvian** | Arquitetura de sistema mais completa da turma: fases 0–4 com *gate* de métrica, máquina de estados, catálogo de métricas, baseline dos 4 lotes medida (296 notas sem foto, 866 órfãs, 1.316 medidores múltiplos). Triagem V0 por heurística (Laplaciano, histograma); V1 = MobileNetV3 multirrótulo com rótulo fraco por degradação sintética | A parte profunda é a menos especificada do documento e está empurrada para a Fase 2. Nenhum modelo treinado. Risco: chegar ao Marco 2 com a plataforma pronta e sem rede |
| **Luminus** (Grupo 2) | Único protótipo rodando em dado real: PP-OCRv5 (detecção) + PP-OCRv6-tiny / TrOCR (reconhecimento). Benchmark no UFPR-AMR: leitura exata 0,357 e acurácia por dígito 0,846. Latência em CPU: média 4,43 s, p90 13,2 s (382 imagens) | Tudo *off-the-shelf*. O *score* de coerência não está calibrado contra decisão real. O benchmark é em dado público, não na distribuidora. O plano de ajuste fino ainda não está especificado |
| **VoltLens** (Grupo 3) | Entendimento do negócio e dos dados concluídos. Objetivo em três *status* (confirmada / divergência / impedimento-inconclusivo) + extração de número do medidor e consumo. Achados próprios: 347 leituras com valor zero, 150–200 fotos por pasta sem registro correspondente | Nenhuma arquitetura ainda. O cronograma fala em "treinamento com algoritmos de ML" sem dizer quais. A meta "assertividade em 50% das fotos legíveis" não está em forma mensurável |
| **Nortdata** ("Projeto IV") | A análise exploratória de imagem mais profunda da turma (metadados, estatísticas de pixel e integridade das 12.340 imagens). Duas arquiteturas em cascata desenhadas: I (pipelines separados por tipo de medidor) e II (multiclasse), com CNN "é medidor?", PDI (histograma, FFT, Laplaciano), homografia, YOLO por campo, OCR e regex. Implantação: Windows local | A premissa "banco de imagens já rotuladas" não se confirmou. Muitos estágios sem conta de propagação de erro. YOLO exige caixas desenhadas e tem questão de licença. As duas variantes não foram comparadas |

Se o grupo discordar desta leitura, a primeira seção do `README.md` é o lugar para dizer — com
evidência.

---

## O que entregar

```
lab-02/
├── README.md            respostas, tabelas e a leitura dos resultados
├── ARQUITETURA.md       a ficha de arquitetura (modelo na seção 14 do guia)
├── lab02.ipynb          notebook executado, saídas visíveis
├── rotulos/
│   ├── esquema.md       definição de cada classe, com 2 exemplos-limite por classe
│   └── rotulos.csv      nome_arquivo, lote, rotulador, <suas colunas de rótulo>
└── USO-DE-IA.md         registro obrigatório
```

Rode do zero antes de entregar: `Reiniciar e executar tudo`. As fotos **não** vão para o
repositório (dado do cliente) — o *notebook* lê de um caminho configurável no topo.

---

## Parte comum — todos os grupos (0,80)

### A. Ficha de arquitetura (0,20)

Preencha `ARQUITETURA.md` seguindo as 14 seções do guia. Três exigências que separam uma
ficha de um texto:

1. **Toda escolha tem uma alternativa descartada e um motivo.** "Usamos MobileNetV3" não
   conta. "MobileNetV3-Small em vez de ResNet-18 porque 0,93 M × 11,7 M parâmetros e a
   implantação é CPU" conta.
2. **A seção 2 separa o que é aprendido do que é regra.** Comparar o valor lido com o valor
   digitado não é tarefa de rede.
3. **A seção 3 tem a conta do produto das taxas por estágio**, estimada se ainda não
   medida — e diz de onde veio a estimativa.

### B. O modelo profundo em código (0,15)

Implemente em PyTorch o **principal bloco profundo** da arquitetura do grupo (a parte
específica abaixo diz qual). Pode partir do esqueleto.

- **B1.** Imprima a **tabela de formas** camada a camada com uma entrada do tamanho real
  que vocês vão usar.
- **B2.** Reporte parâmetros **totais e treináveis**, na configuração de extração e na de
  ajuste fino.
- **B3.** Justifique a **ativação de saída e a perda** de cada cabeça: as classes são
  exclusivas? Por quê? Dê um exemplo real da base que prova a resposta.

### C. Rótulos e partição (0,20)

Não existe rótulo na base. Sem rótulo não há rede, então este é o item mais pesado.

- **C1.** Escreva `rotulos/esquema.md`: definição operacional de cada classe e **dois
  exemplos-limite** por classe (a foto que quase é, e a que quase não é).
- **C2.** Rotule **no mínimo 300 fotos** no esquema do grupo, estratificadas pelos quatro
  lotes. Com 5 a 7 integrantes, são 45–60 fotos por pessoa.
- **C3.** **50 dessas fotos** devem ser rotuladas por **dois integrantes, às cegas**.
  Reporte o **kappa de Cohen** por classe. Kappa abaixo de 0,6 numa classe significa que a
  definição está ambígua: reescreva o esquema e diga o que mudou.
- **C4.** Particione **por lote** (guia, seção 11) e demonstre numericamente: nenhum
  identificador de leitura e nenhum número de medidor aparece em dois conjuntos.

### D. Sanidade e baseline (0,15)

- **D1.** Perda inicial com pesos aleatórios comparada a ln(C) (ou ln 2 por sigmoide).
- **D2.** Sobreajuste de um lote de 10–20 fotos **reais** até perda próxima de zero. Se não
  chegar, diagnostique antes de seguir.
- **D3.** Treine o bloco profundo em modo **extração de características** nos rótulos de C2
  e compare com uma **baseline não profunda** no mesmo conjunto de validação e na mesma
  métrica. Três sementes, média e desvio. Se o modelo profundo perder, diga por que acha que
  perdeu — isso vale nota cheia.

### E. Confiança e recusa (0,10)

- **E1.** Diagrama de confiabilidade e ECE na validação, antes e depois de *temperature
  scaling*.
- **E2.** Curva **cobertura × risco** da saída de decisão do grupo. Marque nela o ponto da
  meta de negócio que o grupo declarou no kickoff, escrita como "cobertura ≥ X com risco
  ≤ Y". Se a meta não cabe na curva, digam.

---

## Parte específica — por grupo (0,20)

Cada grupo responde **apenas o seu bloco**. As quatro tarefas têm o mesmo peso (0,05).

### Korvian — tirar a rede da Fase 2

Bloco profundo da parte B: **a triagem V1** (MobileNetV3 com cabeça multirrótulo de defeitos).

- **K1. Resolução de entrada.** O documento fixa 360 × 480 nativo. Meça latência e métrica
  com entrada nativa e com 224 px. A triagem de **qualidade** precisa da resolução nativa, ou
  só a leitura precisa? Decida com os números.
- **K2. O rótulo fraco se sustenta?** Treine a cabeça de qualidade **só com degradação
  sintética** das fotos boas e avalie **só em fotos reais** rotuladas em C2. Reporte a queda
  por defeito. Defina também a regra para uma foto real já borrada que recebe borrão
  sintético.
- **K3. Heurística V0 como baseline.** O Laplaciano e o histograma são a baseline oficial
  do grupo. Mesma partição, mesma métrica, e a curva cobertura × risco de V0 e V1 no mesmo
  gráfico.
- **K4. Multitarefa ou modelo separado?** O Estágio 1 (tipo de cena) pode compartilhar a
  espinha dorsal com a triagem de qualidade. Decida com parâmetros, latência e o efeito
  medido na métrica de cada cabeça.

### Luminus — de *off-the-shelf* para modelo do domínio

Bloco profundo da parte B: **o reconhecedor de dígitos** do display recortado.

- **L1. Desmontar as duas famílias.** CRNN + CTC (família PP-OCR rec) × codificador ViT +
  decodificador Transformer (TrOCR). Para cada: entrada, formato da saída, perda,
  parâmetros, e **como erra** — mostre 10 erros reais de cada e classifique em "perdeu
  dígito", "duplicou dígito", "trocou dígito", "inventou sequência".
- **L2. O benchmark do cliente.** Construa ≥ 200 recortes de display da distribuidora com a
  leitura transcrita (podem fazer parte dos 300 de C2) e reavalie os modelos neles. Compare
  com o UFPR-AMR e quantifique o *gap* de domínio.
- **L3. Qual métrica otimiza.** Leitura exata (0,357) ou acurácia por dígito (0,846)? Declare
  **uma** otimizadora e pelo menos duas restrições satisfatórias com número (uma delas de
  latência p90). Justifique pelo custo de uma fatura errada.
- **L4. Plano de ajuste fino e de latência.** O que congelar, quais taxas, que aumento de
  dados é permitido (e qual é proibido para dígitos). E: onde estão os 13,2 s do p90 —
  detecção, reconhecimento, ou a correção de rotação? Meça por estágio.

### VoltLens — do objetivo à primeira arquitetura

Bloco profundo da parte B: **um classificador por transferência** "legível / ilegível / não é
medidor" — o primeiro tijolo que qualquer uma das arquiteturas possíveis vai precisar.

- **V1. Três *status* viram arquitetura.** Para cada *status* (confirmada, divergência,
  impedimento-inconclusivo), quais sub-tarefas aprendidas o produzem e onde entra a regra.
  Desenhe duas versões — estágios e ponta a ponta — e escolha com os critérios da seção 3 do
  guia, considerando que não existe rótulo final.
- **V2. A meta em forma de número.** "Assertividade em ao menos 50% das fotos legíveis" vira
  "cobertura ≥ 50% das legíveis com precisão ≥ X% em *Leitura confirmada*". Declare X e
  defenda com o custo de confirmar uma leitura errada.
- **V3. Pareamento antes do rótulo.** Formalize a regra de pareamento foto ↔ registro (as
  150–200 fotos a mais por pasta). Quantas ficam órfãs com a regra de vocês? Rótulo em foto
  mal pareada é rótulo errado.
- **V4. Os 347 zeros como teste difícil.** Separe as leituras com valor zero que têm foto e
  use-as como conjunto de estresse do classificador: o que ele diz delas? Uma foto que mostra
  ~6.990 e foi registrada como 0 é caso de divergência — o seu *pipeline* chegaria lá?

### Nortdata — da cascata desenhada à cascata medida

Bloco profundo da parte B: **a CNN "é medidor? / é legível?"** do topo da cascata, com
espinha dorsal pré-treinada.

- **N1. Arquitetura I × II.** Compare as duas com argumento quantitativo: quantos modelos
  treinar, quantos rótulos cada um exige, parâmetros totais, latência estimada em CPU. Inclua
  o custo de **desenhar caixas** para o YOLO (meça quantos segundos um integrante leva por
  foto em 20 fotos, e extrapole).
- **N2. Orçamento de erro da cascata.** Em 100 fotos rotuladas, meça a taxa de acerto de
  cada estágio que já existir (os de PDI existem hoje) e estime os demais. Calcule a taxa
  ponta a ponta. Qual estágio mais perde? Para onde vão as fotos do ramo "pode melhorar?"
  quando não melhoram?
- **N3. Especificação do detector.** Classes (consumo, número de série, função), resolução
  de entrada (640 × nativa 480), tamanho do modelo, e **licença**: Ultralytics YOLOv8/11 é
  AGPL-3.0. Para um produto entregue ao cliente, qual é a consequência e qual alternativa
  permissiva vocês usariam (SSDlite ou Faster R-CNN do `torchvision`, PP-OCR det)?
- **N4. Restrição Windows local.** Sem GPU presumida: exporte o bloco da parte B para ONNX,
  prove a paridade com o PyTorch (diferença máxima < 1e-4 num lote real) e meça p50 e p95 em
  CPU. A cascata inteira cabe na janela de trabalho dos analistas?

---

## Como a nota é composta

| Dimensão | Peso | O que eu procuro |
|---|:--:|---|
| Execução | 20% | *notebook* roda do zero, saídas visíveis, sanidade de D1–D2 passa |
| Correção técnica | 30% | ativação e perda certas para o tipo de classe, partição sem vazamento, kappa medido, calibração feita na validação |
| Justificativa | 35% | toda escolha com alternativa e número; a parte específica responde ao estado **real** do grupo |
| Comunicação | 15% | ficha legível em 4–6 páginas, diagrama claro, tabelas com unidade |

### Penalizações

| Situação | Efeito |
|---|---|
| *Notebook* não executa | Execução = 0 e −50% nas demais |
| Partição aleatória por foto | −40% em Correção técnica |
| Limiar ou temperatura ajustados no teste | −30% em Correção técnica |
| Softmax aplicada antes de `CrossEntropyLoss`/`BCEWithLogitsLoss` | −20% em Correção técnica |
| Fotos do cliente commitadas no repositório | −30% no total (dado do cliente, LGPD) |
| Sem `USO-DE-IA.md` | nota zero |
| Atraso até 48h | −20% |

### Bônus (até 0,1)

- Medir o volume real com o cliente (mensal × diário, seção 13 do guia) e refazer o
  orçamento de latência com ele.
- Uma curva de dados (métrica × nº de rótulos: 50, 100, 200, 300) que mostre se vale a pena
  rotular mais.

---

## Dicas

1. **Comece por C.** Sem rótulo, B, D e E não saem do lugar. Dividam a rotulagem no primeiro
   dia e combinem o esquema antes.
2. **Os 50 com dupla rotulagem são às cegas de verdade.** Planilhas separadas, sem conversar
   até os dois terminarem.
3. **Leiam o documento do vizinho.** O mesmo problema tem quatro leituras na sala. Citar a
   solução de outro grupo — e dizer por que vocês não a adotaram — conta a favor na
   Justificativa.
4. **Baseline perdendo não é fracasso.** Com 300 rótulos, uma regressão logística sobre
   *features* congeladas pode empatar com o ajuste fino. Reportar isso com três sementes vale
   mais do que um número alto de uma semente só.
5. **Pergunta que a banca vai fazer:** *"por que essa arquitetura, e não uma camada a
   menos?"* A ficha é a resposta escrita.

---

## Modelo de `USO-DE-IA.md`

```markdown
# Uso de IA generativa neste laboratório

| Integrante | Ferramenta | O que pedi | O que aceitei | O que descartei e por quê |
|---|---|---|---|---|
| ... | Claude | revisão do esquema de rótulos | a sugestão de exemplos-limite | a lista de classes: não batia com as notas do cliente |

## Declaração
Cada integrante é capaz de explicar qualquer linha do código entregue.
```
