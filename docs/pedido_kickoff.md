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
| "Número do medidor" com mais de um código (Q5 dos colegas) | **Padrão confirmado com formato `A/B`** (não é erro de digitação): 320 de 3.479 linhas (9,2%) têm dois números separados por `/`, ex. `3245096207/3190410624`. **Testado e refutado:** o número embutido no nome do arquivo de imagem (ex. `..._20000000002952998707_000.jpg`) **não corresponde a nenhum dos dois números do medidor** nem ao valor da leitura — é provavelmente um ID de ordem de serviço/sequência interna, não o serial do medidor. Pergunta ainda aberta: o que representa cada um dos dois números (medidor antigo/novo? titular/UC?) e qual decide a leitura correta. |
| Mais arquivos de imagem na pasta do que linhas no CSV, ou vice-versa (Q9 dos colegas) | **Confirmado, nas duas direções.** 2.992 imagens em disco vs. 2.795 referenciadas por linhas não-NA do CSV → **197 imagens no disco não referenciadas por nenhuma linha**. Todas as 2.795 referências do CSV existem em disco (0 ausentes). Ainda não sabemos a origem das 197 órfãs — perguntar ao cliente. |
| Nota exige foto (SIM) mas coluna Foto = NA | Ainda não cruzado por nota individual — fica como próximo passo de análise (script simples: join do CSV com a planilha de notas, filtrar SIM + Foto=NA). |
| Nomenclatura de pasta/arquivo (Q10 dos colegas) | **Ainda não decodificada.** Pasta `PSP_EXTRATLEITIMPL_030726_0121` — hipótese: `030726` = data (03/07/2026), mas `0121` no fim e o número de 20 dígitos no nome do arquivo (`00000000000211765824` / `20000000002952998707`) não batem com nenhum campo do CSV (nem medidor, nem leitura). **Pergunta genuína pro Kickoff, dado não resolve sozinho.**

## O que o grupo precisa validar (2 camadas) — atualizado

| Camada | Objetivo | Dados atuais |
|---|---|---|
| 1. OCR do display | Ler o número do medidor na foto | ✅ MAPEN/leiturista tem ~3.840 imgs públicas (UFPR-AMR BR) — baseline já rodado (`relatorio_benchmark.md`) |
| 2. Foto ↔ nota | Validar se a foto corresponde ao medidor/cliente e é coerente com a ocorrência | ✅ **RESOLVIDO** — ~12,3k fotos reais + CSV de ocorrência + catálogo de 61 notas com flag de exigência de foto (`data/neoenergia_pe/`) |

**Pendência real agora é rótulo de qualidade/aceitação, não volume de dados**: o CSV traz a nota
aplicada pelo leiturista, mas não traz uma coluna "foto aceita/rejeitada pelo analista" — esse é
o rótulo-ouro que falta para treino supervisionado direto (ver Q2 abaixo).

## Questões a levar ao Kickoff (revisadas — sem repetir o que os dados já respondem)

1. **Rótulo de decisão da fiscalização:** existe (ou pode ser gerado) um campo "foto
   aceita/rejeitada pelo analista" por linha do `BaseExtracao`? Sem isso, o treino supervisionado
   da Tarefa 2 não tem alvo direto.
2. Os quatro lotes que temos (03/07, 20/05, 21/05, 22/05) são representativos do volume mensal
   (50–70 mil) ou são uma amostra reduzida/específica? Dá pra ter acesso a mais lotes/período?
3. **Anonimização/LGPD:** as fotos têm fachada/portão/pessoas visíveis. Qual o fluxo de uso
   autorizado para este material em trabalho acadêmico (e potencial publicação em artigo)?
4. Por que ~73% das linhas com nota NA (leitura normal) também têm foto associada — é
   fotografia de toda visita, ou auditoria por amostragem?
5. O que significam os dois números separados por `/` em "Número do medidor" (320 casos)? Qual
   dos dois é o correto para a leitura?
6. O que representam os segmentos do nome de pasta (`PSP_EXTRATLEITIMPL_<data>_<seq>`) e do
   arquivo de imagem (número de ~20 dígitos) — não correspondem a medidor nem leitura no CSV.
7. As 197 imagens sem referência no CSV (do lote analisado) — são descartes, fotos duplicadas,
   ou erro de extração?
8. Dá para cruzar as fotos com a região (bairro/CEP) para priorizar fiscalização onde a perda de
   energia é alta (integração com a camada de dados do projeto-irmão `mapen`)?

## Questões operacionais/técnicas remanescentes (dos colegas — `perguntas_cliente_colegas.docx`)

Já respondidas pelos dados (ver tabela acima): fluxo de nota NA com foto, medidor com dois
códigos, contagem de imagens vs. CSV, catálogo de notas que exigem foto. **Ainda em aberto:**

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
