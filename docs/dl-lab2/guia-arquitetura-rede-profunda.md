# Como montar a arquitetura de uma rede profunda

**Guia de apoio ao Laboratório 2** · Deep Learning · BD e IA, 4º período · CESAR School
Aplicado ao Projeto 4 (distribuidora — verificação de fotos de leitura)

---

## Para que serve este guia

Os quatro grupos trabalham sobre o mesmo cliente, os mesmos 12.340 JPGs e as mesmas quatro
extrações diárias. Mesmo assim, chegaram ao kickoff com quatro arquiteturas diferentes: um
funil em cascata com YOLO e OCR, um *pipeline* de OCR pronto com *score* de coerência, uma
plataforma faseada em que o modelo entra na Fase 2, e um objetivo de três *status* ainda sem
arquitetura. Nenhuma delas está errada. O que falta, em todas, é o mesmo passo: **justificar
cada bloco da rede com um número ou com uma restrição do problema**.

Este guia descreve esse passo em doze etapas, na ordem em que as decisões dependem umas das
outras. Cada etapa termina com o que precisa aparecer na **ficha de arquitetura** (seção 14),
que é o entregável central do Lab 2.

O código de referência está em [`codigo/esqueleto_arquitetura.py`](codigo/esqueleto_arquitetura.py).
Roda em CPU, sem baixar nada, e todos os números citados abaixo como "medido" saíram dele.

---

## 1. Comece pela decisão, não pela rede

A pergunta "qual rede usar?" só tem resposta depois de outras cinco. Os documentos de
entendimento do negócio de vocês já responderam a maior parte delas — o trabalho aqui é
**copiá-las para a ficha e verificar se a arquitetura obedece a elas**.

| Pergunta | Por que ela restringe a arquitetura | Exemplo no Projeto 4 |
|---|---|---|
| Qual decisão é apoiada? | Define a saída final do sistema | ACEITA / REJEITADA / DÚVIDA por foto |
| Qual a unidade de análise? | Define o que é **uma amostra** — e portanto o que não pode vazar | Uma foto associada a uma leitura (não o medidor, não o cliente) |
| Quem age sobre a saída? | Define se a saída precisa de confiança, motivo ou só rótulo | O analista confere as DÚVIDAS e uma amostra das ACEITAS |
| Qual erro custa mais? | Define a métrica, o limiar e a perda | Aceitar foto inválida (fatura errada) × rejeitar foto boa (revisita) |
| Onde e como roda? | Define o orçamento de parâmetros e latência | Lote pós-coleta, CPU, Windows local ou servidor |

> **Regra:** se uma camada da rede não serve a nenhuma linha desta tabela, ela não deveria
> estar lá.

---

## 2. Decomponha a decisão em tarefas de aprendizado

Uma decisão de negócio quase nunca é **uma** tarefa de aprendizado. "A foto comprova a
leitura?" se decompõe em perguntas menores, cada uma com seu tipo de problema, sua saída e
sua perda. A tabela abaixo é o mapa completo do Projeto 4; nenhum grupo precisa de todas as
linhas.

| Sub-tarefa | Tipo de problema | Saída da rede | Ativação de saída | Perda | Rótulo necessário |
|---|---|---|---|---|---|
| É medidor? Qual cena? | Classificação **exclusiva** | 1 classe entre C | softmax | entropia cruzada | 1 classe por foto |
| Quais defeitos de qualidade? | Classificação **multirrótulo** | 0 a K defeitos simultâneos | sigmoide por defeito | BCE | vetor binário por foto |
| Onde está o display / a placa? | **Detecção** | caixas + classe | por caixa | *box* + classe (YOLO, SSD) | caixas desenhadas |
| Qual número está no display? | **Sequência** | cadeia de dígitos | softmax por passo | **CTC** ou entropia cruzada autoregressiva | transcrição do recorte |
| Ponteiro/disco eletromecânico | **Regressão** (ângulo) | número contínuo | nenhuma (ou seno/cosseno) | MSE / L1 | ângulo ou leitura |
| Foto coerente com a nota (ex. I100)? | Classificação condicionada | sim/não dada a nota | sigmoide | BCE | par (foto, nota) → sim/não |
| Valor lido bate com o digitado? | **Não é aprendizado** | comparação | — | — | — |

A última linha importa. A comparação entre o valor extraído e o valor digitado pelo
leiturista é uma **regra determinística** — colocar uma rede ali é pagar incerteza sem
ganhar nada. A arquitetura de consistência (não OCR pleno) que aparece nos documentos de dois
grupos parte exatamente dessa separação.

**Na ficha:** a lista de sub-tarefas que o grupo vai **aprender**, cada uma com as colunas
acima preenchidas, e a lista das que serão **regra**.

---

## 3. Pipeline em estágios ou modelo ponta a ponta?

Existem dois extremos: uma rede única que recebe a foto e devolve ACEITA/REJEITADA/DÚVIDA, ou
uma cascata de modelos pequenos, cada um especializado. A escolha não é de gosto; é de
**onde estão os rótulos**.

| Critério | Favorece ponta a ponta | Favorece estágios |
|---|---|---|
| Rótulos disponíveis | Muitos pares (foto, decisão final) | Poucos rótulos finais, mas dá para rotular cada etapa |
| Depuração | Difícil: o erro não tem endereço | Fácil: a análise de erro aponta o estágio |
| Dados públicos reaproveitáveis | Raros para a tarefa inteira | Existem por etapa (UFPR-AMR para leitura de display) |
| Latência | Uma passada só | Soma das passadas, mas estágios podem cortar cedo |
| Manutenção | Retreinar tudo | Trocar um estágio sem mexer nos outros |

No Projeto 4 não há rótulo final: nenhuma decisão dos seis analistas foi registrada. Isso
empurra todos os grupos para estágios — e cobra um preço que precisa estar na ficha.

### O preço da cascata: erros se multiplicam

Se uma foto boa precisa passar por quatro estágios para ser ACEITA, a taxa de acerto ponta a
ponta é o **produto** das taxas de cada estágio:

```
0,95 (é medidor) × 0,90 (legível) × 0,85 (achou o display) × 0,80 (leu certo) = 0,58
```

Quatro estágios "bons" entregam 58%. É por isso que a planilha de análise de erro do
Marco 2 tem **uma coluna por estágio**: sem ela, a equipe otimiza o estágio errado.

**Na ficha:** o diagrama do *pipeline*, a taxa estimada (ou medida) de cada estágio e o
produto ponta a ponta.

---

## 4. Defina o contrato de entrada

As fotos do PDA são **360 × 480 px, RGB, 8 bits por canal, retrato (3:4), sem EXIF**. Todas
iguais. Isso é uma vantagem rara — e uma armadilha, porque a resolução é baixa para leitura.

### Quanto de resolução cada tarefa precisa

A MobileNetV3 reduz a imagem por um fator 32. Medido no esqueleto, com a entrada nativa:

```
entrada          (B,   3, 480, 360)
extrator.0       (B,  16, 240, 180)
extrator.2       (B,  24,  60,  45)
extrator.4       (B,  40,  30,  23)
extrator.9       (B,  96,  15,  12)
extrator.12      (B, 576,  15,  12)   ← cada célula resume ~32×30 px da foto
```

- **Classificar a cena** (medidor digital, ciclométrico, portão fechado) é uma decisão sobre
  a imagem inteira. Reduzir para 224 px costuma bastar, e corta a latência quase à metade.
- **Ler dígitos** é outra história. Se o display ocupa um quinto da largura (~72 px) e tem
  cinco ou seis dígitos, cada dígito tem ~12 px. Reduzindo a foto para 224 de largura, cai
  para ~7 px. Por isso toda arquitetura de leitura **recorta primeiro e lê o recorte na
  resolução original** — nunca lê a foto inteira reduzida.

### Regras do contrato

| Item | Decisão a registrar |
|---|---|
| Tamanho | Nativo ou reduzido, **por tarefa**, com a justificativa acima |
| Proporção | Não distorcer 3:4 em 1:1. Use *padding* (*letterbox*) ou redimensione mantendo proporção |
| Normalização | Com backbone pré-treinado no ImageNet: média `[0,485; 0,456; 0,406]`, desvio `[0,229; 0,224; 0,225]` |
| Canais | RGB para cena e qualidade (reflexo tem cor); tons de cinza costuma bastar para o recorte de dígitos |
| Várias fotos por leitura | 4,2% das imagens têm sufixo `_001` a `_008`. Decidir: cada foto é uma amostra, ou a leitura é a amostra? |

**Na ficha:** uma tabela com o contrato de entrada de **cada** modelo.

---

## 5. Escolha a espinha dorsal (*backbone*)

Com poucos rótulos, treinar do zero não é opção: a espinha dorsal vem pré-treinada e o grupo
decide **qual** e **quanto dela ajustar** (Aula 08). O critério é o orçamento de latência e de
memória, não o topo do *ranking* do ImageNet.

| Espinha dorsal | Parâmetros | Quando faz sentido |
|---|--:|---|
| MobileNetV3-Small | 2,5 M (0,93 M sem o classificador) | CPU, Windows local, lote grande, latência apertada |
| MobileNetV3-Large | 5,5 M | Mesmo cenário, com um pouco mais de folga |
| EfficientNet-B0 | 5,3 M | Melhor acurácia por parâmetro; mais lenta em CPU que a MobileNet |
| ResNet-18 | 11,7 M | Referência didática; fácil de inspecionar e de ajustar |
| ConvNeXt-Tiny | 28,6 M | Só se houver GPU na implantação e rótulos suficientes |

O número da MobileNetV3-Small sem classificador foi medido no esqueleto: **932.201**
parâmetros com as duas cabeças do Projeto 4; com o extrator congelado, sobram **5.193**
treináveis. Essa diferença de 180× é a razão pela qual a extração de características funciona
com 200 imagens rotuladas e o ajuste fino completo, não.

**Na ficha:** a espinha dorsal escolhida, a alternativa descartada, e o motivo com número
(parâmetros, latência medida, ou memória).

---

## 6. Desenhe as cabeças e escolha a ativação de saída

A cabeça é a parte da rede que traduz as características em **resposta**. O erro mais comum
da turma, em anos anteriores, é usar softmax onde as classes não são exclusivas.

| As classes são... | Ativação | Perda | Exemplo |
|---|---|---|---|
| Mutuamente exclusivas | softmax | `CrossEntropyLoss` | A foto é de medidor digital **ou** ciclométrico **ou** cena de ocorrência |
| Independentes, podem coexistir | sigmoide por classe | `BCEWithLogitsLoss` | A foto tem reflexo **e** está fora de foco |
| Um número contínuo | nenhuma | `L1Loss` / `MSELoss` | Ângulo do ponteiro |
| Uma sequência de símbolos | softmax por passo | `CTCLoss` | Leitura `04821` do display |

> Em PyTorch, `CrossEntropyLoss` e `BCEWithLogitsLoss` recebem **logits**. Aplicar softmax
> ou sigmoide antes da perda é um erro silencioso: treina, mas pior.

### Multitarefa: uma espinha dorsal, duas cabeças

Cena e qualidade olham para a mesma foto e precisam das mesmas características de baixo
nível (bordas, brilho, textura). Compartilhar a espinha dorsal economiza uma passada inteira
e funciona como regularização (Caruana, 1997). No esqueleto:

```python
class ModeloTriagem(nn.Module):
    def __init__(self, pretreinado=False, p_dropout=0.2):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretreinado else None)
        self.extrator = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cabeca_cena = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(576, 4))       # softmax
        self.cabeca_qualidade = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(576, 5))  # sigmoide

    def forward(self, x):
        z = self.pool(self.extrator(x)).flatten(1)
        return self.cabeca_cena(z), self.cabeca_qualidade(z)
```

A perda total é a soma ponderada: `CE(cena) + λ · BCE(qualidade)`. O peso `λ` é
hiperparâmetro; comece com 1 e só mude se uma das tarefas estagnar enquanto a outra melhora.

**Quando não compartilhar:** se uma tarefa precisa de resolução muito diferente da outra (ler
dígitos × classificar cena), ou se os dados de treino de uma são dez vezes maiores que os da
outra.

**Na ficha:** para cada cabeça, número de saídas, ativação, perda e o argumento de exclusividade.

---

## 7. Trate o desbalanceamento na perda, não só na amostragem

A base é muito desigual: 66,6% das linhas sem nota, T181 sozinho é 56,9% das notas, 19 dos 61
códigos aparecem. Nos defeitos de qualidade, a distribuição ainda não é conhecida — e será
desigual.

| Técnica | Como | Quando |
|---|---|---|
| Peso por classe | `CrossEntropyLoss(weight=...)`, peso ∝ 1/frequência | Classes exclusivas desbalanceadas |
| `pos_weight` | `BCEWithLogitsLoss(pos_weight=neg/pos)` por defeito | Multirrótulo com defeito raro |
| *Focal loss* | reduz o peso dos exemplos fáceis (Lin et al., 2017) | Muitos exemplos fáceis dominando o gradiente |
| Reamostragem | `WeightedRandomSampler` | Quando a classe rara tem tão poucos exemplos que quase não aparece num lote |

Qualquer uma delas **descalibra** a probabilidade de saída. Isso não é problema se a etapa 9
(calibração) for feita depois — é problema se o grupo usar a probabilidade crua como "confiança".

**Na ficha:** a distribuição de rótulos medida na amostra rotulada e a técnica escolhida.

---

## 8. Ajuste a capacidade ao número de rótulos

Não existe rótulo pronto. Todos os grupos vão rotular à mão, então a pergunta útil é: **com
N rótulos, quanto da rede eu posso ajustar?**

| Rótulos por classe | Estratégia (Aula 08) | Parâmetros treináveis no esqueleto |
|---|---|--:|
| Até ~50 | Extração de características: extrator congelado, treina só a cabeça | 5.193 |
| ~50 a ~500 | Ajuste fino dos últimos blocos, taxa menor no extrator | dezenas a centenas de milhares |
| Mais de ~500 | Ajuste fino completo, com taxas diferenciadas por bloco | 932.201 |

Os limites são ordens de grandeza, não leis. A curva de dados do Lab 4 é o que diz onde está
o ponto de virada **no domínio de vocês**.

Taxas diferenciadas, em código:

```python
opt = torch.optim.AdamW([
    {"params": modelo.extrator.parameters(), "lr": 1e-4},
    {"params": modelo.cabeca_cena.parameters(), "lr": 1e-3},
    {"params": modelo.cabeca_qualidade.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
```

**Na ficha:** quantos rótulos o grupo terá, qual estratégia isso permite, e quantos parâmetros
ficam treináveis.

---

## 9. Projete a saída com confiança e com direito a recusar

Todos os grupos definiram, de algum jeito, uma terceira saída além de sim e não: DÚVIDA,
"impedimento / inconclusivo", "requer conferência", "fila humana". Essa saída é a parte mais
importante da arquitetura — é ela que protege a fatura. Ela tem três requisitos técnicos.

**a) Probabilidade calibrada.** Uma rede que diz 0,95 precisa acertar ~95% das vezes em que
diz 0,95. Redes modernas são sistematicamente confiantes demais (Guo et al., 2017). Meça com
o **diagrama de confiabilidade** e o **ECE** na validação; corrija com *temperature scaling*
(um único parâmetro `T` ajustado na validação: `softmax(z / T)`).

**b) Regra de recusa explícita.** No esqueleto:

```python
def decidir(logits_cena, logits_qual, lim_cena=0.90, lim_defeito=0.50, lim_limpo=0.10):
    # REJEITADA: cena confiável e (não é medidor, ou tem defeito provável)
    # ACEITA:    cena confiável e nenhum defeito acima do limiar de limpeza
    # DÚVIDA:    todo o resto
```

Com pesos aleatórios, as oito fotos de teste saem **todas DÚVIDA** — é o comportamento certo
de um sistema que ainda não sabe nada.

**c) Curva cobertura × risco.** Varie o limiar e plote, para cada valor, a **cobertura**
(fração de fotos que o sistema decide sozinho) contra o **risco** (taxa de erro nas que
decidiu). A meta de negócio vira um ponto nessa curva: "decidir 30–40% do volume" (Korvian),
"processar com assertividade 50% das legíveis" (VoltLens). Sem a curva, essas metas são
desejos (Geifman & El-Yaniv, 2017).

**Na ficha:** a regra de decisão, onde os limiares serão escolhidos (validação, nunca teste),
e a meta traduzida em "cobertura ≥ X com risco ≤ Y".

---

## 10. Escolha o aumento de dados que respeita o domínio

Aumento de dados não é neutro: ele declara o que **não muda o rótulo**. No Projeto 4, várias
transformações comuns mudam.

| Transformação | Pode usar? | Por quê |
|---|---|---|
| Espelhamento horizontal | **Não**, para leitura | `2` espelhado não é dígito; `04821` vira outra sequência |
| Rotação pequena (±10°) | Sim | O leiturista inclina o aparelho |
| Brilho, contraste, cor | Sim, com moderação | Varia com sol, sombra, hora |
| Desfoque gaussiano | **Depende da tarefa** | Para leitura, é aumento. Para a cabeça de qualidade, **troca o rótulo** para "fora de foco" |
| Reflexo sintético | Mesmo caso do desfoque | É aumento para cena, é rótulo para qualidade |
| Recorte aleatório | Com cuidado | Pode cortar o display e tornar a foto "enquadramento incorreto" |

A linha do desfoque é o centro da estratégia de rótulo fraco de um dos grupos: degradar
fotos boas para fabricar exemplos de defeito. Funciona, **se** o grupo provar que a rede
aprendeu o defeito e não o artefato da degradação sintética. O teste é simples e obrigatório:
treinar só com sintético e medir numa amostra **real** rotulada à mão.

**Na ficha:** a lista de transformações por modelo, com a coluna "muda o rótulo?".

---

## 11. Particione pelo que pode vazar

A unidade de análise é a foto, mas a unidade de **vazamento** é maior: fotos da mesma leitura
(`_000`, `_001`...) são quase idênticas, e o mesmo medidor pode reaparecer em outra extração.

Medido nas quatro extrações (`NE-csv/`), contando números de medidor em comum entre lotes:

| Par de lotes | Medidores em comum |
|---|--:|
| 20/05 × 21/05 | 4 |
| 20/05 × 22/05 | 3 |
| 21/05 × 22/05 | 3 |
| 03/07 × qualquer outro | 0 a 2 |

Cada lote tem de 3.600 a 3.935 medidores distintos. A sobreposição é desprezível: **partir
por lote é seguro e resolve o vazamento por medidor de graça**. Com quatro lotes, a sugestão
é:

| Conjunto | Lote | Por quê |
|---|---|---|
| Treino | 20/05 e 21/05 | Dois dias consecutivos, maior volume |
| Validação | 22/05 | Mesmo mês, dia seguinte: mede generalização entre dias |
| Teste | 03/07 | Seis semanas depois: o mais próximo de "produção" que a base permite |

Se o grupo misturar dados públicos (UFPR-AMR) com a base da distribuidora, precisa do conjunto
**treino-dev** (Capítulo 5 da apostila): uma fatia do treino, da mesma distribuição, que
separa variância de descasamento de distribuição.

**Na ficha:** a partição, a demonstração numérica de que nenhuma leitura e nenhum medidor
aparece em dois conjuntos, e a justificativa.

---

## 12. Verifique a sanidade antes de treinar de verdade

Antes de gastar uma hora de GPU, quatro verificações que custam segundos (Karpathy, 2019).
Todas estão no esqueleto, com a saída real:

| Verificação | O que esperar | Medido no esqueleto |
|---|---|---|
| **Formas** em cada camada | Nenhuma surpresa; a tabela da seção 4 | ok |
| **Perda inicial** com pesos aleatórios | ≈ ln(C) para softmax; ≈ ln 2 para cada sigmoide | CE = 1,384 (ln 4 = 1,386); BCE = 0,696 (ln 2 = 0,693) |
| **Sobreajustar um lote** de 4 a 20 exemplos | Perda vai a ~0. Se não vai, há defeito na rede, na perda ou nos rótulos | triagem: 2,08 → 0,009 em 60 passos; CRNN: 12,2 → 0,003 em 300 passos |
| **Baseline não profunda** no mesmo teste | O modelo profundo precisa ganhar dela, ou o grupo explica por que não | — (é tarefa do grupo) |

A perda inicial longe de ln(C) quase sempre significa inicialização errada da última camada,
ou uma softmax aplicada antes da `CrossEntropyLoss`.

### O caso especial da leitura: CTC

O leitor de dígitos do esqueleto (`CRNNDigitos`, Shi et al., 2017) reduz a altura do recorte
a 1 e usa a largura como eixo de tempo: 32 passos para um recorte de 32 × 128 px. A perda CTC
(Graves et al., 2006) dispensa dizer onde está cada dígito — só a sequência. Ela introduz um
símbolo "branco", e é esse símbolo que permite ler `00029` e `69900`: sem ele, dígitos
repetidos colapsariam em um só. No sobreajuste do esqueleto, as quatro leituras de teste
(`04821`, `17413`, `00029`, `69900`) saem idênticas ao alvo.

Os grupos que usam TrOCR estão no outro ramo: codificador ViT + decodificador Transformer
autoregressivo (Li et al., 2023). Ele não precisa de CTC, é mais forte em texto variado — e
pode **inventar** uma sequência plausível que não está na imagem. Para dígitos de medidor,
esse modo de falha é mais perigoso que o do CTC, que tende a perder ou duplicar um dígito.

**Na ficha:** a saída das quatro verificações, coladas do *notebook*.

---

## 13. Feche o orçamento de latência e de implantação

A arquitetura só está pronta quando cabe no lugar onde vai rodar. O orçamento é uma conta:

```
volume diário × latência por foto ≤ janela de processamento disponível
```

Os grupos registraram volumes que não batem entre si, e isso muda a conta por um fator 20:

| Fonte | Volume declarado | Por dia (≈22 dias úteis) |
|---|---|--:|
| Luminus, VoltLens, Nortdata | 50–70 mil fotos por **mês** | 2.300–3.200 |
| Korvian (doc. de entendimento) | 30–50 mil fotos por **dia** | 30.000–50.000 |
| Os próprios lotes recebidos | ~3.100 imagens por extração | 3.100 |

Com a latência medida pela Luminus (OCR pronto em CPU, **média 4,43 s, p90 13,2 s**, 382
imagens), 3.200 fotos levam ~4 h de CPU — cabe num lote noturno. 50.000 fotos levariam
~61 h — não cabe em um dia com uma máquina. **Qual volume é o real é pergunta para o cliente,
e a resposta decide a espinha dorsal.** Registrem a hipótese adotada.

Para CPU, o caminho padrão é exportar para ONNX e rodar com ONNX Runtime, verificando a
**paridade**: a diferença máxima entre as saídas do PyTorch e do ONNX num lote real precisa
ficar abaixo de 1e-4. Uma exportação sem teste de paridade não é uma exportação.

**Licença também é restrição de implantação.** As versões mais usadas do YOLO (Ultralytics
v8/11) são AGPL-3.0: embarcá-las num produto entregue a um cliente tem consequência de
licença. Alternativas permissivas: SSDlite/Faster R-CNN do `torchvision` (BSD), PP-OCR
(Apache-2.0), TrOCR (MIT).

**Na ficha:** volume adotado, latência medida por foto (p50 e p95), tempo total do lote,
onde roda, formato de exportação e licença de cada componente.

---

## 14. A ficha de arquitetura

É um arquivo `ARQUITETURA.md` no repositório do grupo. Cada seção corresponde a uma etapa
deste guia. Uma ficha boa cabe em quatro a seis páginas; uma ficha de vinte páginas
geralmente esconde que nenhuma decisão foi tomada.

```markdown
# Ficha de arquitetura — <nome do grupo>

## 1. Decisão apoiada
Decisão, unidade de análise, quem age, erro mais caro, onde roda.

## 2. Sub-tarefas
| Sub-tarefa | Aprendida ou regra? | Tipo | Saída | Ativação | Perda | Rótulo |

## 3. Pipeline
Diagrama. Taxa estimada/medida por estágio. Produto ponta a ponta.

## 4. Contrato de entrada (por modelo)
| Modelo | Tamanho | Proporção | Normalização | Canais |

## 5. Espinha dorsal
Escolhida, descartada, motivo com número.

## 6. Cabeças
| Cabeça | Nº saídas | Ativação | Perda | Classes exclusivas? Por quê |

## 7. Desbalanceamento
Distribuição medida na amostra rotulada. Técnica.

## 8. Capacidade × rótulos
Nº de rótulos. Estratégia de transferência. Parâmetros treináveis / totais.

## 9. Confiança e recusa
Regra de decisão. Calibração. Meta como "cobertura ≥ X com risco ≤ Y".

## 10. Aumento de dados
| Transformação | Modelo | Muda o rótulo? |

## 11. Partição
Conjuntos. Prova numérica de ausência de vazamento.

## 12. Sanidade
Formas, perda inicial, sobreajuste de um lote, baseline.

## 13. Implantação
Volume adotado. Latência p50/p95. Onde roda. Exportação. Licenças.

## 14. O que decidimos não fazer
E por quê.
```

---

## 15. Anti-padrões que aparecem todo semestre

| Anti-padrão | Sintoma | Correção |
|---|---|---|
| Escolher a rede pelo *ranking* | "Usamos a maior porque é a melhor" | Orçamento de latência e de rótulos primeiro (seções 5 e 13) |
| Softmax em classes não exclusivas | A foto com reflexo **e** desfoque só pode ter um defeito | Sigmoide + BCE (seção 6) |
| Probabilidade crua como confiança | "0,99 de confiança" que erra 10% das vezes | Calibração + diagrama de confiabilidade (seção 9) |
| Rede onde cabe regra | Um modelo para comparar dois números | Regra determinística (seção 2) |
| Ler a foto inteira reduzida | Dígitos com 7 px | Detectar, recortar, ler na resolução nativa (seção 4) |
| Partição aleatória por foto | `_000` no treino, `_001` no teste | Partição por lote (seção 11) |
| Métrica por dígito como métrica final | 0,85 de acurácia por dígito, 0,36 de leitura exata | Leitura exata: um dígito errado é uma fatura errada |
| Cascata sem conta de propagação | Cada estágio "funciona", o sistema não | Produto das taxas + análise de erro por estágio (seção 3) |

---

## Referências

- CARUANA, R. Multitask Learning. *Machine Learning*, v. 28, p. 41–75, 1997.
- GEIFMAN, Y.; EL-YANIV, R. Selective Classification for Deep Neural Networks. In: *NeurIPS*, 2017.
- GRAVES, A. et al. Connectionist Temporal Classification: labelling unsegmented sequence data with recurrent neural networks. In: *ICML*, 2006.
- GUO, C. et al. On Calibration of Modern Neural Networks. In: *ICML*, 2017.
- HOWARD, A. et al. Searching for MobileNetV3. In: *ICCV*, 2019.
- KARPATHY, A. *A Recipe for Training Neural Networks*. 2019. Disponível em: karpathy.github.io/2019/04/25/recipe.
- LAROCA, R. et al. Convolutional Neural Networks for Automatic Meter Reading. *Journal of Electronic Imaging*, v. 28, n. 1, 2019. (conjunto UFPR-AMR)
- LI, M. et al. TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. In: *AAAI*, 2023.
- LIN, T.-Y. et al. Focal Loss for Dense Object Detection. In: *ICCV*, 2017.
- SHI, B.; BAI, X.; YAO, C. An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN). *IEEE TPAMI*, v. 39, n. 11, 2017.
- Apostila da disciplina: Capítulo 4 (métrica otimizadora e restrições), Capítulo 5 (cadeia de suposições, treino-dev, análise de erro), Capítulos 7 e 8 (CNNs e transferência).
