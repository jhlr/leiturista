# Pedido de dados e questões para o Kickoff 12/09 — distribuidora parceira

**Data:** 2026-08-08 (rev. 2026-08-25) | **Status:** GAP DA TAREFA 2 RESOLVIDO — dados reais em mãos
**Referência:** `projeto4_desafio.md` (desafio), `analise_suficiencia.md` (gap, **desatualizado**),
`sbti_artigo/entendimento_problema.docx`, `data/distribuidora_campo/` (amostra real)
**Meta:** confirmar no Kickoff os pontos que a amostra real não responde sozinha.

> **ATUALIZAÇÃO CRÍTICA (2026-08-25):** o grupo já tem em mãos uma amostra real de
> **uma distribuidora parceira** (`data/distribuidora_campo/`, gitignored): 4 lotes diários
> (`PSP_EXTRATLEITIMPL_030726_0121`, `_200526_0352`, `_210526_0335`, `_220526_0408`) somando
> **12.340 imagens de campo** (13.669 linhas de CSV) + 4 CSVs `BaseExtracao_<data>_Dia.csv` + a planilha
> `DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx` com o **catálogo completo de 61
> códigos de ocorrência** e a coluna "NOTA EXIGE FOTO?" (32 SIM / 29 NÃO). Isso **invalida o "GAP
> CRÍTICO: nenhum dataset" registrado em `analise_suficiencia.md` e `relatorio_benchmark.md` §6**
> — ambos precisam de revisão. Mantive o histórico deste doc abaixo, mas a seção "Achados a
> partir da amostra real" substitui boa parte do que antes eram perguntas em aberto.

---

## Achados a partir da amostra real (`data/distribuidora_campo/`, analisado 2026-08-25)

**Reproduzível:** `scripts/distribuidora_stats.py` — agrega os 4 lotes diários (13.669 linhas de CSV,
12.340 imagens em disco) e recalcula tudo abaixo automaticamente.

| Pergunta original | Resposta encontrada nos dados (agregado dos 4 lotes) |
|---|---|
| Catálogo de códigos de ocorrência e padrão esperado de foto (Q3 do grupo; Q8 dos colegas) | **Resolvida.** `DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx` lista as 61 notas com descrição e flag SIM/NÃO de exigência de foto (32 SIM / 29 NÃO). A nota mais frequente é `T181` "Função Não Existe no Sistema" (2.595 ocorrências), seguida de `P111` "Medidor Substituído" (685) e `L101` "Leitura Informada Pelo Cliente" (496). Nenhum código de imóvel fechado/demolido (I1xx/D1xx) apareceu nos 4 lotes disponíveis — a amostra pode não ser representativa de todo o espectro de ocorrências. |
| Por que "Nota de Leitura Atual" = NA aparece com foto associada (Q6 dos colegas) | **Padrão confirmado, motivo ainda não explicado pelos dados.** 7.265 das 9.109 linhas com nota NA (**79,8%**) têm foto associada — leitura normal também gera foto na grande maioria dos casos, não é exclusivo de ocorrência. Perguntar ao cliente **por que** (fotografia de toda visita por padrão? auditoria por amostragem?). |
| "Número do medidor" com mais de um código (Q5 dos colegas) | **Padrão confirmado com formato `A/B`** (não é erro de digitação): 1.316 de 13.669 linhas (**9,6%**) têm dois números separados por `/`. **Testado e refutada a hipótese óbvia:** só 48 dos 1.316 casos (3,6%) têm nota `P111` (medidor substituído) — a maioria (762, 58%) está em linhas com nota `NA` (leitura normal), então o padrão **não é primariamente troca de medidor**. O número embutido no nome do arquivo de imagem não corresponde a nenhum dos dois números nem ao valor da leitura. Pergunta que sobra pro Kickoff: o que cada número representa e qual decide a leitura correta. |
| Mais arquivos de imagem na pasta do que linhas no CSV, ou vice-versa (Q9 dos colegas) | **Confirmado nos 4 lotes.** Em cada lote há entre 197 e 242 imagens em disco sem nenhuma linha do CSV que as referencie (866 no total das 4). Todas as referências do CSV existem em disco (0 ausentes, nos 4 lotes). Pergunta que sobra: são descartes do app (foto refeita) ou uma etapa de exportação que perde a referência? |
| Nota exige foto (SIM) mas coluna Foto = NA | **Cruzamento feito nos 4 lotes.** Das 3.943 linhas com nota que exige foto (SIM), **296 (7,5%) não têm foto** — concentradas em `T181` (189 casos), `P111` (58) e `L101` (41). É uma minoria concentrada em 3 códigos, não um problema sistêmico da planilha. Pergunta que sobra: por que essas linhas especificamente não geraram foto apesar da nota exigir. |
| Nomenclatura de pasta/arquivo (Q10 dos colegas) | **Parcialmente decodificada.** Nome de pasta `PSP_EXTRATLEITIMPL_<data>_<seq>` — `<data>` é plausivelmente a data de extração. O ID de 20 dígitos no nome do arquivo tem duas famílias distintas de formato, nenhuma batendo com "Número do medidor" nem com a leitura do CSV. **Pergunta genuína pro Kickoff, dado não resolve sozinho:** o que esse ID representa e por que existem dois formatos. |
| **Sinais de coerência do pipeline `leiturista` batem com a semântica da nota, sem rótulo?** | **Testado empiricamente** com `scripts/distribuidora_pipeline_eval.py` — rodamos o MeterOCR já existente em amostras reais de `NA` (leitura normal), `T111` (caixa/tampa danificada — obstrução relatada) e `L101` (leitura informada pelo cliente, presumivelmente ilegível em campo). Resultado no `docs/sbti_artigo/artigo_sbti2026.docx` §4.3 e no detalhe em `data/distribuidora_campo/pipeline_eval.json` (gitignored — contém número de medidor e nome de arquivo reais, não commitar). |

## O que o grupo precisa validar (2 camadas) — atualizado

| Camada | Objetivo | Dados atuais |
|---|---|---|
| 1. OCR do display | Ler o número do medidor na foto | ✅ MAPEN/leiturista tem ~3.840 imgs públicas (UFPR-AMR BR) — baseline já rodado (`relatorio_benchmark.md`) |
| 2. Foto ↔ nota | Validar se a foto corresponde ao medidor/cliente e é coerente com a ocorrência | ✅ **RESOLVIDO** — ~12,3k fotos reais + CSV de ocorrência + catálogo de 61 notas com flag de exigência de foto (`data/distribuidora_campo/`) |

**Pendência real agora é rótulo de qualidade/aceitação, não volume de dados**: o CSV traz a nota
aplicada pelo leiturista, mas não traz uma coluna "foto aceita/rejeitada pelo analista" — esse é
o rótulo-ouro que falta para treino supervisionado direto (ver Q2 abaixo).

## Questões a levar ao Kickoff (só o que os dados NÃO resolveram sozinhos)

1. **Rótulo de decisão da fiscalização:** existe (ou pode ser gerado) um campo "foto
   aceita/rejeitada pelo analista" por linha do `BaseExtracao`? Sem isso, o treino supervisionado
   da Tarefa 2 não tem alvo direto — é a única lacuna de dado real que sobra.
2. Os quatro lotes que temos (03/07, 20/05, 21/05, 22/05) são representativos do volume mensal
   (50–70 mil) ou são uma amostra reduzida/específica? Dá pra ter acesso a mais lotes/período?
3. **Anonimização/LGPD:** as fotos têm fachada/portão/pessoas visíveis. Qual o fluxo de uso
   autorizado para este material em trabalho acadêmico (e potencial publicação em artigo)?
4. O que significam os dois números separados por `/` em "Número do medidor" (320 casos, 9,2% do
   lote) — não é majoritariamente troca de medidor (só 4,4% tem nota P111). Qual dos dois decide
   a leitura?
5. O que representa o ID de 20 dígitos no nome do arquivo de imagem (duas famílias de formato,
   nenhuma bate com medidor/leitura do CSV) e o sufixo `_0121` no nome da pasta?
6. As 197 imagens sem referência no CSV (do lote analisado) — descarte do app, foto refeita, ou
   perda de vínculo na exportação?
7. Por que os 40 casos (de 927) com nota que exige foto acabam sem foto registrada?
8. Dá para cruzar as fotos com a região (bairro/CEP) para priorizar fiscalização onde a perda de
   energia é alta (integração com a camada de dados do projeto-irmão `mapen`)?

## Questões operacionais/técnicas remanescentes (dos colegas — `perguntas_cliente_colegas.docx`)

Já respondidas pelos dados (ver tabela acima): nota NA com foto, medidor com dois códigos,
contagem de imagens vs. CSV, catálogo de notas que exigem foto, cruzamento SIM×Foto=NA.
**Genuinamente sem resposta nos dados que temos — só o cliente sabe:**

1. Como as imagens são baixadas para a máquina local? Com que frequência? Em lotes — de que tamanho?
2. O que faz uma foto ser rejeitada (ilegível, fora de contexto) e o que acontece na planilha de
   controle quando isso ocorre? (a coluna de decisão não está no CSV que temos — ver Q1 acima)
3. Após conferência, como a informação é consolidada no SAP? Sobe um CSV — é o mesmo
   `BaseExtracao_<data>_Dia.csv` ou outro template?
4. Nome correto do departamento/cargo envolvido na conferência de imagens?
5. Qual o modelo do leitor PDA usado em campo?

## Plano B — não se aplica mais para volume/catálogo

Mantido apenas para o rótulo de decisão (aceita/rejeitada), caso o cliente não consiga fornecer:

1. **Piloto manual:** rotular à mão uma amostra dos ~12,3k já disponíveis (aceita/rejeitada por
   coerência com a nota) para treinar/validar um classificador inicial.
2. Priorizar as notas que **exigem foto** (32 códigos, lista acima) — maior densidade de sinal
   de coerência foto↔ocorrência do que notas sem exigência de foto.
