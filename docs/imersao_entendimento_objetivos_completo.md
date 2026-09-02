# Imersão, Entendimento e Objetivos do Problema — versão completa

| Campo | Conteúdo |
|---|---|
| Projeto | Análise automatizada de evidências fotográficas de leitura (distribuidora de energia elétrica) |
| Grupo | Grupo 2 — Luminus |
| Integrantes | Angelita Dias, Carolline Mariz, Fernando Rangel, Guaraci Rios, Ivo Caetano, João Rietra, Mikael Mulatinho |
| Data | 2026-08-31 |
| Status | Rascunho para revisão do grupo antes do Kickoff (12/09) |

Versão completa do documento de imersão do grupo, preenchida com os dados reais já levantados
sobre o problema. Trechos ainda sem confirmação do cliente ficam marcados
**[A CONFIRMAR NO KICKOFF]**.

---

## 1. Problema, decisão e ação

**Qual problema organizacional justifica o projeto?**

A distribuidora recebe entre 50 mil e 70 mil fotografias por mês como evidência de leituras com
crítica de valor, analisadas manualmente por apenas 6 analistas/validadores. O volume gera carga
operacional elevada e horas extras, e reduz o tempo disponível da equipe para os casos que
efetivamente exigem julgamento especializado. O pedido de origem — "automatizar a fiscalização das
fotos" — não trazia um critério objetivo do que torna uma foto suficiente/coerente, nem o ponto do
fluxo em que a triagem deveria acontecer.

**Qual decisão será apoiada?**

Decidir, para cada fotografia enviada à fila de validação, se a evidência é suficiente e coerente
com a ocorrência aplicada pelo leiturista — ACEITA, REJEITADA ou DÚVIDA (encaminhar para o
analista).

**Em qual momento a decisão ocorre?**

Entre a geração da foto em campo (quando o dispositivo aponta crítica no valor) e a consolidação
no SAP/faturamento — ou seja, antes de a foto entrar na fila de conferência manual dos 6 analistas.

**Qual é a unidade de análise?**

Uma fotografia associada a uma leitura — não o processo de leitura como um todo, nem o cliente.
Uma mesma visita com crítica de valor pode gerar mais de uma foto, e cada foto recebe seu próprio
score de coerência.

**Qual output deverá ser produzido?**

Por fotografia: a classificação ACEITA, REJEITADA ou DÚVIDA, com o score de coerência, o motivo
(legibilidade insuficiente, leitura não detectada, incoerência com a nota) e a leitura do número
do display quando aplicável. Faltando confiança suficiente em algum dos sinais, o sistema declara
DÚVIDA e não decide sozinho — nunca estima.

**Qual ação humana ou operacional será realizada?**

O analista/validador confere as fotos sinalizadas como DÚVIDA (e uma amostra das ACEITA, por
auditoria) e decide: aceitar — a foto segue para o faturamento — ou rejeitar — tratativa adicional.
A decisão final permanece humana nos casos de dúvida.

**Qual resultado organizacional é esperado?**

Reduzir o volume de fotos que passam por conferência manual integral, liberando capacidade da
equipe de 6 analistas para os casos que efetivamente exigem julgamento especializado, e reduzindo
a necessidade de horas extras.

## Imersão do problema

A distribuidora é responsável pela distribuição de energia elétrica em praticamente todo o
estado. Começou como estatal, foi privatizada em 2000 e integrada a um grupo global de energia;
adotou o nome atual em 2021. Atende milhões de clientes e investe em modernização das redes,
digitalização e sustentabilidade.

A operação de leitura e entrega de contas envolve, na sua grande maioria, visitas às unidades
consumidoras para o registro da leitura dos medidores — atividade estratégica para o caixa da
empresa (receita registrada e cobrada de forma transparente) e para o relacionamento com o
consumidor. Durante a visita, caso o dispositivo utilizado pelo leiturista para registrar a
leitura do medidor realize alguma crítica referente ao valor registrado, o leiturista deve
providenciar uma fotografia do medidor para servir como evidência do registro realizado. Esses
registros integram o processo de fiscalização e contribuem para a confiabilidade das informações
utilizadas no faturamento.

Parte dessas fotografias, fruto de possíveis inconsistências no valor registrado, são
encaminhadas para validação. Atualmente esse volume varia entre **50 mil e 70 mil imagens por
mês**, analisadas manualmente por uma equipe de **seis analistas/validadores**, que verificam se
a evidência apresentada é suficiente e compatível com o registro efetuado pelo leiturista. O
elevado número de análises aumenta a carga operacional da equipe e contribui para a realização de
horas extras.

O grupo já obteve uma amostra real da distribuidora parceira: 4 lotes diários de campo, somando
**12.340 imagens** e 13.669 linhas de planilha de ocorrência, além de um catálogo completo de
**61 códigos de ocorrência** aplicados pelo leiturista, dos quais **32 exigem foto e 29 não
exigem**. A nota mais frequente na amostra é `T181` — "Função Não Existe no Sistema" (2.595
ocorrências), seguida de `P111` — "Medidor Substituído" (685) e `L101` — "Leitura Informada Pelo
Cliente" (496). Nenhum código de imóvel fechado/demolido (família I1xx/D1xx) apareceu nos 4 lotes
disponíveis — a amostra pode não ser representativa de todo o espectro de ocorrências que o
cliente enfrenta no volume mensal completo.

O recorte deste projeto compreende a análise das evidências fotográficas associadas às visitas de
leitura. Não integram esse recorte o cálculo tarifário, a manutenção da rede elétrica, a
investigação de fraude, a cobrança financeira ou outras atividades não diretamente relacionadas à
validação dessas evidências.

### Stakeholders

| Stakeholder | Papel / interesse | Responsabilidade / situação |
|---|---|---|
| Leiturista (campo) | Registra a leitura no dispositivo móvel e fotografa o medidor quando o sistema aponta crítica no valor. Aplica o código de ocorrência que descreve o motivo da crítica. | Fonte primária do dado; fora do escopo do projeto alterar seu fluxo de trabalho em campo. |
| Analista/validador | Usuário final da solução. Confere manualmente a foto e decide aceitar ou rejeitar; hoje faz isso para 50–70 mil imagens/mês, equipe de 6 pessoas. | A automação deve reduzir volume repetitivo sem remover a decisão final sobre casos de dúvida. |
| Distribuidora (cliente) | Patrocinadora do desafio. Interessada em reduzir custo/tempo de fiscalização e em manter a confiabilidade do faturamento. | Fornece a amostra de dados reais e valida critérios de sucesso; nome do interlocutor formal **[A CONFIRMAR NO KICKOFF]**. |
| Consumidor final | Impactado indiretamente: erro de leitura não detectado chega à fatura; foto rejeitada indevidamente pode gerar visita extra. | Não interage diretamente com o projeto, mas é o parâmetro final de qualidade. |
| Equipe de TI / dados da distribuidora | Especialista de sistema — mantém o `BaseExtracao` e a integração com o SAP; possível fonte do rótulo de decisão (aceita/rejeitada) que hoje falta na base. | Consultada no projeto; papel a confirmar **[A CONFIRMAR NO KICKOFF]**. |

### Escopo

| Categoria | Definição |
|---|---|
| Incluído | Análise das evidências fotográficas associadas a visitas de leitura com crítica de valor; verificação de legibilidade do display; verificação de coerência entre a foto e o código de ocorrência aplicado pelo leiturista; sinalização de casos de dúvida para análise humana. |
| Excluído | Cálculo tarifário; manutenção da rede elétrica; investigação de fraude; cobrança financeira; decisão de aceite/rejeição sem revisão humana disponível — o projeto sinaliza, o analista decide. |
| Dependências | Rótulo real de "foto aceita/rejeitada pelo analista" (hoje ausente na base recebida); confirmação do volume/representatividade da amostra frente ao total mensal; posição da distribuidora sobre uso e anonimização das fotos (fachada, portão, eventuais pessoas visíveis) em trabalho acadêmico. |

### Fluxo atual do processo (as-is)

O leiturista visita a unidade e registra a leitura no dispositivo. Se o dispositivo aponta crítica
no valor, o leiturista fotografa o medidor e aplica um código de ocorrência; a foto e a ocorrência
entram na planilha de controle e caem na fila de validação (50 a 70 mil imagens/mês), onde os 6
analistas conferem manualmente, uma a uma, se a evidência é suficiente e coerente — só então o
resultado (aceito ou rejeitado) é consolidado no SAP/faturamento. Quando não há crítica, a leitura
segue direto para o faturamento, sem passar por essa fila. O gargalo do processo atual está
concentrado exatamente nessa conferência manual: todo o volume passa por 6 pessoas, sem nenhuma
triagem automática prévia.

### Fluxo proposto com automação (to-be)

A proposta introduz um pipeline automatizado entre a geração da foto e a fila de conferência: a
foto passa por verificação de legibilidade, leitura automática do número do display e verificação
de coerência com a ocorrência aplicada, gerando um score de confiança. Casos de alta confiança
(consistentes) seguem para auditoria por amostragem, em vez de conferência integral; casos de
baixa confiança ou dado insuficiente vão para uma fila priorizada, onde o analista decide como
hoje. A proposta não elimina o analista humano — introduz uma triagem automática antes da fila,
de forma que a capacidade da equipe se concentre nos casos de dúvida real, exatamente o recorte
que os objetivos específicos OE4 e OE5 (abaixo) descrevem.

## Entendimento e descrição do problema

O problema central está na necessidade de avaliar um elevado volume de fotografias sinalizadas
como potencialmente inconsistentes em relação ao registro da leitura do medidor. Atualmente,
entre 50 mil e 70 mil imagens demandam análise, realizada predominantemente de forma manual por
seis analistas/validadores, que verificam individualmente se a evidência fotográfica é suficiente
e coerente com o registro efetuado pelo leiturista.

A questão operacional do projeto pode, portanto, ser sintetizada da seguinte forma: **a evidência
fotográfica apresenta qualidade suficiente e é coerente com o tipo de medidor e com a leitura
realizada?** Na prática, o problema se desdobra em duas subquestões distintas, confirmadas na
amostra real:

1. **Legibilidade/qualidade da foto** — o display está nítido o suficiente para conferência? Um
   sinal simples de nitidez (variância do Laplaciano do recorte do display, abaixo de um limiar
   indicando "display borrado") já foi testado contra o rótulo real de um dataset público de
   medidores: nitidez isolada **não** prediz se a leitura automática do número acerta ou erra
   (correlação praticamente nula, poder discriminativo próximo do acaso) — é um sinal fraco
   sozinho, não um classificador de aceite/rejeição.
2. **Coerência foto ↔ ocorrência** — a foto corresponde ao que a nota do leiturista alega? Testado
   empiricamente agrupando as 61 notas em 10 grupos por semelhança semântica: o sinal "a leitura
   automática do número não foi detectada na foto" discrimina seletivamente — sobe bastante em
   notas de semântica visual/física (ex.: medidor com defeito físico, leitura informada pelo
   cliente, obstrução — entre 52% e 73% dos casos) mas cai **abaixo** da taxa da nota neutra `NA`
   (sem ocorrência, ~37%) em notas puramente administrativas (ex.: função inexistente no sistema,
   ~25%). Isso é evidência de que o sinal carrega informação real sobre a ocorrência, e não é
   ruído genérico do pipeline de leitura.

O volume de verificações manuais faz com que profissionais especializados dediquem parte
significativa de sua jornada a tarefas repetitivas, contribuindo para a realização de horas
extras e reduzindo o tempo disponível para casos que efetivamente exigem análise especializada.
O projeto busca, assim, tornar o processo de conferência de imagens mais eficiente e padronizado,
diferenciando registros suficientemente consistentes daqueles que apresentam dúvida, ambiguidade
ou potencial inconsistência.

Não se pressupõe que todas as decisões possam ser realizadas sem intervenção humana. Pretende-se
alcançar o maior nível de automatização tecnicamente seguro e operacionalmente confiável,
preservando a atuação dos analistas nas situações em que a evidência seja insuficiente ou o
julgamento especializado permaneça necessário.

### Arquitetura de referência do pipeline de análise

![Arquitetura do pipeline — detecção, leitura, classificação e sinais de coerência](/private/tmp/claude-501/-Users-joaorietra-Developer-leiturista/c565faab-2c87-467f-b567-efe7801ac05e/scratchpad/mermaid/arquitetura_pipeline.png){width=6in}

O protótipo já implementado segue quatro etapas em sequência: (1) detecta as regiões de texto na
foto; (2) lê o conteúdo de cada região (OCR); (3) classifica cada leitura como "número do display"
ou "número de série / placa"; (4) combina três sinais — legibilidade, leitura detectada ou não, e
identidade do medidor (via série/placa) — num único score de coerência entre a foto e a ocorrência
aplicada. Hoje esse score ainda não tem calibração contra o rótulo real de aceite/rejeição do
analista — é a peça central que falta confirmar no Kickoff.

### Catálogo de ocorrências — recorte da amostra analisada

| Código | Descrição | Ocorrências na amostra | Exige foto? |
|---|---|---:|:---:|
| `T181` | Função não existe no sistema | 2.595 | Não |
| `P111` | Medidor substituído | 685 | Sim |
| `L101` | Leitura informada pelo cliente | 496 | Sim |
| `NA` | Sem ocorrência (leitura normal) | 9.109 (66,7% do total) | — |
| *(demais 57 códigos)* | Cobrem o restante da amostra; família de imóvel fechado/demolido (I1xx/D1xx) não apareceu nos 4 lotes analisados | — | 32 de 61 códigos exigem foto |

**O que a amostra real já revelou e que precisa de confirmação do cliente:**

- Falta o rótulo-ouro **"foto aceita/rejeitada pelo analista"** na planilha de ocorrência — sem
  ele não há alvo direto para treino supervisionado da tarefa de coerência.
- 79,8% das linhas com nota `NA` (leitura normal, sem ocorrência) têm foto associada — a
  fotografia não é exclusiva de ocorrência; motivo ainda não confirmado com o cliente.
- 9,6% das linhas trazem dois números no campo "Número do medidor" (formato `A/B`); só 3,6%
  desses casos têm nota de troca de medidor (`P111`) — a maioria não é explicada por essa
  hipótese óbvia.
- 7,5% das notas que **exigem** foto (296 de 3.943) aparecem sem foto associada, concentradas em
  3 códigos (`T181`, `P111`, `L101`).
- Latência medida do pipeline de leitura off-the-shelf, rodando em CPU (382 imagens testadas):
  média de 4,43 s/imagem, mediana de 1,4 s, p90 de 13,2 s — referência inicial para dimensionar
  throughput de produção.

## Objetivos do projeto

**Objetivo geral:** desenvolver uma abordagem para tornar a análise das evidências fotográficas
das visitas de leitura mais eficiente, padronizada e confiável, distinguindo registros
consistentes daqueles que necessitem de análise humana, com o propósito de reduzir a carga
operacional e as horas extras dos analistas/validadores.

**Objetivos específicos (OE):**

- **OE1** — **[A CONFIRMAR NO KICKOFF]** não definido no documento de origem do grupo; candidato
  natural pela ordem lógica do problema seria estabelecer a leitura automática do número do
  display como pré-requisito de qualquer validação de coerência — já há um baseline
  off-the-shelf medido em dataset público de medidores brasileiros: ~36% de leituras 100%
  corretas e ~85% de acerto por dígito, sem nenhum treino específico ainda. Não incluir na
  numeração oficial sem validar com o grupo.
- **OE2** — Estabelecer critérios de coerência entre a evidência fotográfica e as informações
  registradas pelo leiturista. *(status: catálogo de 61 códigos de ocorrência com flag de
  exigência de foto em mãos; sinal empírico de coerência já testado agrupando as notas por
  semelhança semântica — falta o rótulo de decisão do analista para calibrar o limiar de aceite.)*
- **OE3** — **[A CONFIRMAR NO KICKOFF]** não definido no documento de origem do grupo; candidato
  natural seria garantir a identidade do medidor entre foto e cadastro do cliente
  (reidentificação por características visuais do próprio medidor — número de série e outros
  textos da placa — já implementada como heurística de protótipo). Não incluir na numeração
  oficial sem validar com o grupo.
- **OE4** — Distinguir registros com evidências suficientemente consistentes daqueles que
  apresentem dúvida, ambiguidade ou potencial inconsistência, direcionando para análise humana os
  casos que efetivamente necessitem de julgamento especializado. *(status: sinais de legibilidade
  e de leitura-não-detectada já implementados e testados; falta rótulo real para transformar em
  classificador calibrado — sem ele, o sistema não deve declarar aceite/rejeição, apenas
  sinalizar.)*
- **OE5** — Reduzir a carga de verificações manuais repetitivas e a necessidade de horas extras,
  permitindo o redirecionamento da capacidade dos analistas/validadores para atividades de maior
  valor operacional. *(status: sem baseline de tempo médio de conferência humana por foto —
  perguntar ao cliente no Kickoff para poder estimar o ganho.)*

## Target, erros e baseline

| Elemento | Definição |
|---|---|
| Target proposto | `foto_coerente`: por imagem, ACEITA, REJEITADA ou DÚVIDA (encaminhar para analista), com o score de coerência e o motivo (legibilidade insuficiente, leitura não detectada, incoerência com a nota). Rótulo de referência a obter da decisão real do analista — hoje ausente na base recebida. |
| Falso positivo | Sistema classifica como ACEITA uma foto que o analista teria rejeitado. Se não houver auditoria por amostragem, o erro chega ao faturamento sem revisão — é o erro mais caro, análogo ao risco que hoje só o analista humano contém. |
| Falso negativo | Sistema classifica como DÚVIDA/REJEITADA uma foto que era, de fato, válida. Consome tempo do analista sem necessidade — custo operacional, não custo de faturamento; erro mais tolerável que o falso positivo na fase inicial. |
| Baseline não-ML | A conferência manual que os 6 analistas fazem hoje é a referência de qualidade e de custo, não uma alternativa hipotética. Candidato a baseline automatizado simples: regra determinística sobre legibilidade (nitidez) + presença de leitura detectada, sem os sinais de coerência semântica por tipo de ocorrência. Os dois precisam ser medidos sobre o mesmo lote rotulado. |

## Critérios de sucesso e restrições

| Dimensão | Critério | Indicador / pendência |
|---|---|---|
| Negócio | Reduzir o volume de fotos que passam por conferência integral, mantendo a decisão final com o analista nos casos de dúvida. | Proporção de fotos resolvidas por auditoria por amostragem vs. conferência integral. Métrica e limiar a definir com a distribuidora. |
| Operacional | Entregar o score de coerência no formato e no momento em que o analista já trabalha, com fundamentação suficiente para decidir rápido nos casos de dúvida. | Tempo médio de conferência por foto (hoje sem baseline medido) e taxa de concordância entre score e decisão do analista. |
| Dados / ML | Igualar ou superar a conferência manual no lote de homologação, sem nunca estimar dado ausente — declarar DÚVIDA em vez de arriscar um falso positivo. | Métricas por classe (ACEITA, REJEITADA, DÚVIDA) e tolerância a falso positivo a definir após obter o rótulo real de decisão. |
| Econômico | Justificar o custo de operação do pipeline frente às horas extras evitadas na equipe de 6 analistas. | Custo por imagem processada contra horas de analista economizadas. Método de mensuração **[A CONFIRMAR NO KICKOFF]**. |

### Riscos e restrições iniciais

| Risco | Impacto | Tratamento inicial |
|---|---|---|
| Alucinação na leitura do número | Erro de OCR com a mesma fluência do acerto pode virar leitura errada aceita silenciosamente. | Exigir que o pipeline declare DÚVIDA quando a confiança da leitura for baixa, em vez de estimar; nunca aceitar automaticamente sem sinal de alta confiança em pelo menos dois sinais independentes (legibilidade + leitura detectada). |
| Falta de rótulo real de decisão | Sem o campo "aceita/rejeitada pelo analista", não há como calibrar limiares nem medir precisão/recall reais — todo o trabalho de modelagem fica bloqueado nesse ponto. | Priorizar essa pergunta no Kickoff (12/09); plano B é rotular manualmente uma amostra da própria distribuidora enquanto o pedido formal tramita. |
| Exposição de dado pessoal nas fotos | As fotos de campo podem mostrar fachada, portão, e eventualmente pessoas — dado sensível sob a LGPD, especialmente em uso acadêmico/artigo. | Tratar o fluxo de anonimização e o uso autorizado como pendência a resolver com a distribuidora antes de qualquer publicação; não redistribuir a amostra bruta fora do grupo. |
| Amostra não representativa | Os 4 lotes disponíveis (12.340 imagens) não trazem nenhuma ocorrência da família I1xx/D1xx (imóvel fechado/demolido) — o comportamento do modelo nesses casos é desconhecido. | Pedir mais lotes/período no Kickoff; não generalizar conclusões da amostra atual para o volume mensal completo (50–70 mil) sem essa confirmação. |
| Agência excessiva | Uso do score de coerência como decisão final de faturamento sem nenhuma auditoria humana, mesmo por amostragem. | Manter a auditoria por amostragem como controle obrigatório sobre os casos de "aceite automático"; a decisão de DÚVIDA sempre vai para o analista. |

## 5. Project Vision v0

| Campo | Definição |
|---|---|
| Problema | 50 a 70 mil fotos/mês de evidência de leitura passam por conferência manual integral de apenas 6 analistas, sem nenhuma triagem automática prévia — carga operacional elevada e horas extras. |
| Decisão | Se a fotografia é suficiente e coerente com a ocorrência aplicada pelo leiturista — ACEITA, REJEITADA ou DÚVIDA. |
| Momento | Entre a geração da foto em campo e a consolidação no SAP/faturamento, antes da fila de conferência manual. |
| Unidade de análise | Uma fotografia associada a uma leitura — não o processo, nem o cliente. |
| Target proposto | `foto_coerente`: ACEITA, REJEITADA ou DÚVIDA, com score de coerência e motivo. |
| Ação | O analista confere os casos de DÚVIDA (e uma amostra dos ACEITA) e decide aceitar ou rejeitar; a decisão final permanece humana. |
| Resultado | Reduzir o volume de conferência manual integral e as horas extras, concentrando a equipe nos casos de dúvida real. |
| Fora do escopo | Cálculo tarifário, manutenção de rede, investigação de fraude, cobrança financeira; decisão de aceite/rejeição sem revisão humana disponível — o projeto sinaliza, o analista decide. |
| Critérios de sucesso | A definir e aprovar antes do gate — nenhum limiar quantitativo foi acordado com a distribuidora. |

## Gate da fase

**Pergunta de decisão:** o problema de decisão está claro e é valioso o suficiente para justificar
o Entendimento dos Dados?

| | Critério | Situação | Evidência ou pendência |
|---|---|---|---|
| [x] | Problema e valor organizacional definidos | Concluído | Problema descrito a partir da amostra real: 50–70 mil fotos/mês, 6 analistas, sem triagem automática prévia. |
| [x] | Decisão, momento e ação definidos | Concluído | Triagem proposta entre a foto e a fila de conferência; ação humana de aceite/rejeição mantida para os casos de dúvida. |
| [x] | Unidade de análise e target propostos | Concluído | Unidade fixada na fotografia; target com três resultados (ACEITA, REJEITADA, DÚVIDA). |
| [ ] | Critérios de sucesso mensuráveis aprovados | Pendente | Nenhum limiar quantitativo definido; tolerância a falso positivo não acordada com a distribuidora. |
| [ ] | Responsáveis e aprovador identificados | Pendente | Interlocutor formal da distribuidora e da equipe de TI ainda não confirmados. |
| [x] | Escopo, restrições e riscos registrados | Concluído | Recorte do projeto definido; riscos de LGPD e de governança registrados. |

### Decisão do gate

| Campo | Conteúdo |
|---|---|
| Resultado | Prosseguir condicionalmente para o Entendimento dos Dados. |
| Condições ou retrabalho | Antes da aprovação final: (i) definir com a distribuidora os critérios quantitativos de sucesso e a tolerância a falso positivo; (ii) confirmar o rótulo real de "foto aceita/rejeitada pelo analista"; (iii) obter posição da distribuidora sobre anonimização e uso das fotos; (iv) confirmar a representatividade da amostra frente ao volume mensal completo. |
| Aprovador | A definir pelo grupo/distribuidora. |
| Data da decisão | A definir (alvo: Kickoff, 12/09). |

## Fontes

LAROCA, R. et al. Deep Learning for Image-based Automatic Dial Meter Reading: Dataset and
Baselines. In: INTERNATIONAL JOINT CONFERENCE ON NEURAL NETWORKS (IJCNN), 2020. Anais [...].
Glasgow: IEEE, 2020. DOI: 10.1109/IJCNN48605.2020.9207318.

LI, M. et al. TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. In:
AAAI CONFERENCE ON ARTIFICIAL INTELLIGENCE, 37., 2023. Anais [...]. Washington, DC: AAAI, 2023.
arXiv:2109.10282.

LIAO, M. et al. Real-Time Scene Text Detection with Differentiable Binarization. In: AAAI
CONFERENCE ON ARTIFICIAL INTELLIGENCE, 34., 2020. Anais [...]. New York: AAAI, 2020.
arXiv:1911.08947.

PADDLEPADDLE. PaddleOCR: modelos de detecção e reconhecimento de texto em cena. Repositório de
código. Disponível em: https://github.com/PaddlePaddle/PaddleOCR. Acesso em: 25 ago. 2026.
