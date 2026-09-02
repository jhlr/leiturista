# Projeto 4 — Desafio de distribuidoras de energia elétrica (verificação de fotos de leitura)

**Disciplina:** Projeto 4 - DADOS (Cesar School, BD2026.2)
**Cliente:** distribuidora de energia elétrica
**Grupo:** Grupo 2 — Luminus (Angelita Dias, Carolline Mariz, Fernando Rangel, Guaraci Rios,
Ivo Caetano, João Rietra, Mikael Mulatinho)
**Última atualização:** 2026-08-31

---

## Sobre a disciplina

Grupos investigam um problema do contexto de uma **distribuidora de energia elétrica** e
propõem/desenvolvem uma
solução aplicada, acompanhada pelo **CRISP-DM**. Disciplinas envolvidas: **Deep
Learning** (Vitinho), **Visão Computacional** (Eron) e **ML Ops** (Galindo).

### Método: ciclos inspirados no CRISP-DM

1. Entendimento do problema
2. Entendimento dos dados
3. Preparação dos dados
4. Modelagem
5. Avaliação
6. Entrega/MVP

### Marcos da disciplina

| Marco | Data | Conteúdo |
|-------|------|----------|
| **Kickoff** | 12/09 | Validação do problema, dados, escopo e plano inicial |
| **SR1** | 03/10 | Validação do dataset, pipeline, baseline e primeiros experimentos |
| **SR2** | 05/12 | Solução final, avaliação, MVP e aprendizados |

### Comunicação

- Slack: `#projeto-4-bd-26-2`
- Plano de Ensino: no Classroom
- Uso responsável de IA: política disponibilizada na disciplina

### Entregáveis da Semana 02

1. **Definição do grupo** — nome, integrantes, email.
2. **Google Site do grupo** — site criado, página inicial organizada, estrutura mínima
   para registrar o andamento (sitemap: Home, SR1, SR2, Lições Aprendidas, Grupo).
3. **Papéis dos integrantes** — papel principal de cada pessoa + mapeamento dos Pontos
   Focais.
4. **Ferramenta de gestão do projeto** — ex.: Trello, Taiga, Notion, GitHub Projects.
5. **Método de pesquisa** — ex.: desk research, análise documental, entrevistas,
   questionário, benchmark, exploração de bases públicas.
6. **Ferramenta de gestão de conhecimento** — ex.: Miro, FigJam/Figma, Notion, Obsidian.
7. **Cronograma preliminar CRISP-DM** — semanas × etapas.

### Papéis principais (sugeridos)

Gestão do projeto; Pesquisa e entendimento do problema; Dados e preparação do dataset;
Modelagem / IA; Desenvolvimento do MVP; Documentação e apresentação.

**Ponto Focal:** representante do grupo no acompanhamento semanal com o professor. Papel
**rotativo semanal** — todos assumem, ninguém repete enquanto houver quem ainda não foi.
No acompanhamento: apresenta entregáveis/avanços da semana, impedimentos + soluções
propostas e panorama de riscos/oportunidades (mini pitch oral).

### Orientação do professor

> "O objetivo não é começar com a solução pronta. O objetivo é começar com clareza,
> organização e método."

---

## O desafio do cliente (distribuidora)

### Contexto da empresa

A distribuidora é responsável por levar energia elétrica a praticamente todo o estado.
Começou como estatal, foi privatizada em 2000 e integrada a um grupo global de energia;
adotou o nome atual em 2021. Atende milhões de clientes e investe em modernização das
redes, digitalização e sustentabilidade.

A área de **leitura e entrega de contas** visita praticamente todos os clientes
mensalmente, garantindo medição correta e fatura entregue — atividade estratégica para o
caixa da empresa (receita registrada e cobrada de forma transparente) e para o
relacionamento com o consumidor.

### Problema/desafio

**Fiscalização das fotos tiradas pelos leituristas em campo.** As imagens comprovam a
leitura realizada, mas o volume é enorme e dificulta verificar todas com atenção.
Problemas gerados:

1. **Controle de qualidade:** nem sempre é possível confirmar se a foto corresponde
   exatamente ao medidor do cliente.
2. **Tempo e custo:** revisar manualmente milhares de fotos consome recursos e atrasa
   processos.
3. **Risco de inconsistências:** sem fiscalização eficiente podem ocorrer erros de
   leitura ou questionamentos dos clientes — impacta a confiança do consumidor e o caixa
   da empresa.

**Tempo de ocorrência:** sempre (contínuo). **Tentativas anteriores de solução:** não.

**Volume e equipe (confirmado pelo grupo, doc. "Imersão, Entendimento e Objetivos do Problema",
2026-08-31):** entre **50 mil e 70 mil imagens/mês** são encaminhadas para validação (fotografias
associadas a inconsistências no valor registrado pelo leiturista), analisadas manualmente por
**6 analistas/validadores**, que verificam se a evidência é suficiente e compatível com o registro.
O volume gera carga operacional elevada e horas extras para a equipe.

**Recorte do projeto:** análise das evidências fotográficas associadas às visitas de leitura.
Fora do escopo: cálculo tarifário, manutenção da rede elétrica, investigação de fraude, cobrança
financeira ou outras atividades não diretamente relacionadas à validação dessas evidências.

**Não se pressupõe automação total:** o objetivo é o maior nível de automatização tecnicamente
seguro e operacionalmente confiável, preservando a atuação dos analistas nos casos de evidência
insuficiente ou que exijam julgamento especializado — a decisão final permanece humana.

### Quem é afetado

- **Clientes** (mais afetados): o erro chega diretamente na fatura.
- **Empresa:** imagem/credibilidade e carga de trabalho da equipe (atravessa toda a
  cadeia de valor da leitura e entrega de contas).

### Impacto nos objetivos

- **Estratégicos:** fortalecer a confiança dos clientes e manter credibilidade; erros
  nas contas comprometem a imagem, geram perda de receita e vão contra a meta de ser uma
  distribuidora moderna, transparente e eficiente.
- **Operacionais:** falta de fiscalização aumenta retrabalho, custos extras e reduz a
  produtividade da equipe de leitura.

### Ideia de solução (apontada pelo cliente)

Automatização com IA — **não apenas validar se a foto corresponde ao medidor e à leitura
correta, mas também se ela está coerente com a nota (ocorrência) aplicada pelo
leiturista**. Exemplos:

- Verificar se a imagem está **legível** e corresponde ao **tipo de ocorrência**
  registrada.
- Se o leiturista aplica a nota **I100 – Casa fechada**, a foto precisa seguir um padrão
  esperado (ex.: portão ou fachada fechada visível).
- A IA valida a leitura correta **e** se a evidência fotográfica está de acordo com a
  justificativa registrada.

---

## Objetivos do projeto (grupo, doc. "Imersão, Entendimento e Objetivos do Problema", 2026-08-31)

**Objetivo geral:** desenvolver uma abordagem para tornar a análise das evidências fotográficas
das visitas de leitura mais eficiente, padronizada e confiável, distinguindo registros
consistentes daqueles que necessitem de análise humana, com o propósito de reduzir a carga
operacional e as horas extras dos analistas/validadores.

**Objetivos específicos** (numeração do grupo — "revisar com base no escopo dos dados"):

- **OE2** — Estabelecer critérios de coerência entre a evidência fotográfica e as informações
  registradas pelo leiturista.
- **OE4** — Distinguir registros com evidências suficientemente consistentes daqueles que
  apresentem dúvida, ambiguidade ou potencial inconsistência, direcionando para análise humana os
  casos que efetivamente necessitem de julgamento especializado.
- **OE5** — Reduzir a carga de verificações manuais repetitivas e a necessidade de horas extras,
  permitindo o redirecionamento da capacidade dos analistas/validadores para atividades de maior
  valor operacional.

*(OE1 e OE3 não aparecem no documento de origem — provavelmente pendentes de definição pelo
grupo; confirmar antes de citar a lista como completa.)*

---

## Relação com este repositório (MAPEN)

Este subprojeto de visão computacional (leitura automática do display do medidor) se
alinha diretamente ao desafio: o MAPEN já reúne os datasets de imagem de medidores
(`data/ufpr_amr/`, `data/praekelt_meter_readings/`, `data/utility_meters_*`, etc.)
para OCR/leitura do display. O desafio da distribuidora adiciona uma segunda camada:
**validação da coerência foto ↔ ocorrência registrada** (além da leitura do número).

Ponto de atenção: os datasets do MAPEN são de medidores residenciais/smart meters
(imagens de display), enquanto o desafio real exige também validar padrões de foto por
tipo de ocorrência (ex.: portão fechado para I100) — sinalizar no planejamento da
disciplina.
