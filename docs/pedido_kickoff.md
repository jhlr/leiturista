# Pedido de dados e questões para o Kickoff 12/09 — Neoenergia PE

**Data:** 2026-08-08 (rev. 2026-08-25) | **Status:** GAP DA TAREFA 2 RESOLVIDO — dados reais em mãos
**Referência:** `projeto4_desafio.md` (desafio), `analise_suficiencia.md` (gap, **desatualizado**),
`sbti_artigo/entendimento_problema.docx`, `data/neoenergia_pe/` (amostra real)
**Meta:** confirmar no Kickoff os pontos que a amostra real não responde sozinha.

> **ATUALIZAÇÃO CRÍTICA (2026-08-25):** o grupo já tem em mãos uma amostra real da
> **Neoenergia PE** (`data/neoenergia_pe/`, gitignored): 4 lotes diários
> (`PSP_EXTRATLEITIMPL_030726_0121`, `_200526_0352`, `_210526_0335`, `_220526_0408`) somando
> **~12.343 imagens de campo** + 4 CSVs `BaseExtracao_<data>_Dia.csv` + a planilha
> `DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx` com o **catálogo completo de 61
> códigos de ocorrência** e a coluna "NOTA EXIGE FOTO?" (32 SIM / 29 NÃO). Isso **invalida o "GAP
> CRÍTICO: nenhum dataset" registrado em `analise_suficiencia.md` e `relatorio_benchmark.md` §6**
> — ambos precisam de revisão. Mantive o histórico deste doc abaixo, mas a seção "Achados a
> partir da amostra real" substitui boa parte do que antes eram perguntas em aberto.

---

## Achados a partir da amostra real (`data/neoenergia_pe/`, analisado 2026-08-25)

Inspeção do lote `PSP_EXTRATLEITIMPL_030726_0121` (3.479 linhas de CSV, 2.992 imagens em disco)
respondeu diretamente várias perguntas que estavam em aberto:

| Pergunta original | Resposta encontrada nos dados |
|---|---|
| Catálogo de códigos de ocorrência e padrão esperado de foto (Q3 do grupo; Q8 dos colegas) | **Resolvida.** `DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO.xlsx` lista as 61 notas com descrição e flag SIM/NÃO de exigência de foto. Ex. de SIM: I100/I110/I120 (local fechado ocupado/desocupado/veraneio), D100/D101/D110/D111 (demolido), P111 (medidor substituído), T181 (função não existe no sistema — a nota mais frequente no lote, 591 ocorrências). |
| Por que "Nota de Leitura Atual" = NA aparece com foto associada (Q6 dos colegas) | **Padrão confirmado, motivo ainda não explicado pelos dados.** 1.758 das 2.393 linhas com nota NA (73%) têm foto associada — leitura normal também gera foto em boa parte dos casos, não é exclusivo de ocorrência. Perguntar ao cliente **por que** (auditoria por amostragem? toda leitura fotografa?). |
| "Número do medidor" com mais de um código (Q5 dos colegas) | **Padrão confirmado com formato `A/B`** (não é erro de digitação): 320 de 3.479 linhas (9,2%) têm dois números separados por `/`, ex. `3245096207/3190410624`. **Testado e refutada a hipótese óbvia:** só 14 dos 320 casos (4,4%) têm nota `P111` (medidor substituído) — a maioria (190, 59%) está em linhas com nota `NA` (leitura normal), então o padrão **não é primariamente troca de medidor**. O número embutido no nome do arquivo de imagem (ex. `..._20000000002952998707_000.jpg`) **não corresponde a nenhum dos dois números** nem ao valor da leitura. Pergunta que sobra pro Kickoff: o que cada número representa (duas unidades consumidoras no mesmo ponto? medidor + caixa de proteção?) e qual decide a leitura correta. |
| Mais arquivos de imagem na pasta do que linhas no CSV, ou vice-versa (Q9 dos colegas) | **Confirmado, nas duas direções.** 2.992 imagens em disco vs. 2.795 referenciadas por linhas não-NA do CSV → **197 imagens no disco não referenciadas por nenhuma linha**. Todas as 2.795 referências do CSV existem em disco (0 ausentes). As 197 órfãs seguem o **mesmo formato de nome e o mesmo prefixo `2...` de ID** que as referenciadas (não são um lote corrompido nem de outra origem) — o mecanismo de geração é o mesmo, só falta o vínculo com uma linha do CSV. Pergunta que sobra: são descartes do app (foto refeita) ou uma etapa de exportação que perde a referência? |
| Nota exige foto (SIM) mas coluna Foto = NA | **Cruzamento feito.** Das 927 linhas com nota que exige foto (SIM), **40 (4,3%) não têm foto** — concentradas em `T181` "Função Não Existe no Sistema" (27 casos), `P111` "Medidor Substituído" (8), `L101` "Leitura Informada Pelo Cliente" (4) e `M101` (1). É uma minoria, não um problema sistêmico da planilha. Pergunta que sobra: por que essas 40 especificamente não geraram foto apesar da nota exigir. |
| Nomenclatura de pasta/arquivo (Q10 dos colegas) | **Parcialmente decodificada.** Nome de pasta `PSP_EXTRATLEITIMPL_030726_0121`: `030726` é plausivelmente a data (03/07/2026). O ID de 20 dígitos no nome do arquivo tem **duas famílias distintas**: 99,8% dos arquivos (2.985 de 2.992) começam com `2` seguido de zeros e um número interno; os outros 7 começam com `0` seguido de um número de 9 dígitos "limpo" (ex. `211770854`) — mesmo formato do número de medidor. Nenhuma das duas famílias bate com "Número do medidor" nem com a leitura do CSV correspondente. **Pergunta genuína pro Kickoff, dado não resolve sozinho:** o que esse ID representa (matrícula do PDA? ordem de serviço? sequência de captura?) e por que existem dois formatos.

## O que o grupo precisa validar (2 camadas) — atualizado

| Camada | Objetivo | Dados atuais |
|---|---|---|
| 1. OCR do display | Ler o número do medidor na foto | ✅ MAPEN/leiturista tem ~3.840 imgs públicas (UFPR-AMR BR) — baseline já rodado (`relatorio_benchmark.md`) |
| 2. Foto ↔ nota | Validar se a foto corresponde ao medidor/cliente e é coerente com a ocorrência | ✅ **RESOLVIDO** — ~12,3k fotos reais + CSV de ocorrência + catálogo de 61 notas com flag de exigência de foto (`data/neoenergia_pe/`) |

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
