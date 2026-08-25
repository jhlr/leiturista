# Pedido de dados e questões para o Kickoff 12/09 — distribuidora

**Data:** 2026-08-08 (rev. 2026-08-25) | **Status:** rascunho pré-kickoff (validar com o professor/grupo)
**Referência:** `projeto4_desafio.md` (desafio), `analise_suficiencia.md` (gap),
`sbti_artigo/entendimento_problema.docx` (documento formal do Grupo 2/Luminus, 2026-08-25)
**Meta:** sair do Kickoff com acesso a um lote real de fotos de campo da distribuidora.

> **Nota (2026-08-25):** o doc `Imersão, Entendimento e Objetivos do Problema` já formalizou
> volume (50–70 mil imagens), equipe (6 analistas), tipos de ocorrência que impedem leitura
> convencional (imóvel fechado/desocupado/demolido, acesso impedido) e a existência de geração
> própria (fotovoltaico). As questões abaixo foram revisadas para não repetir o que já está
> respondido e focar no que ainda falta.
>
> **Achado importante (2026-08-25):** `perguntas_cliente_colegas.docx` (perguntas listadas por
> colegas do grupo) mostra que já houve contato com uma amostra real de artefatos operacionais —
> nome de arquivo `BaseExtracao_"data"_Dia.csv` com colunas (Número do medidor, Posição do medidor
> lida, Nota de Leitura Atual, Foto do medidor), planilha `DESCRIÇÃO NOTAS LEITURISTAS X
> SOLICITAÇÃO DE FOTO`, e convenção de nomes de pasta/arquivo de imagem (`PSP_EXTRATLEITIMPL_
> <data>_<seq>` e `..._<matrícula/serial>_000`). Isso é mais concreto do que o "gap total de
> dataset" assumido em `analise_suficiencia.md` — **confirmar no Kickoff se isso é uma amostra já
> recebida (e então localizá-la no projeto) ou apenas material de referência mostrado em aula**,
> antes de reafirmar o gap como bloqueante.

---

## O que o grupo precisa validar (2 camadas)

| Camada | Objetivo | Dados atuais |
|---|---|---|
| 1. OCR do display | Ler o número do medidor na foto | ✅ MAPEN tem ~3.840 imgs públicas (UFPR-AMR BR, goodcoffee, utility_meters…) — suficiente p/ baseline |
| 2. Foto ↔ nota | Validar se a foto corresponde ao medidor/cliente e é **coerente com a ocorrência** registrada pelo leiturista (ex.: **I100 Casa fechada** → foto deve mostrar portão/fachada fechada) | ❌ **nenhum** dataset público com rótulo de ocorrência |

**A Camada 2 é o coração do desafio e não temos dados.** Por isso o pedido abaixo.

## Pedido de dados (lote real de campo)

### 1. Fotografias de campo com nota/ocorrência (CRÍTICO — Camada 2)

Lote de fotos tiradas por leituristas, **com a ocorrência aplicada**, idealmente:

- **Volume:** qualquer lote ajuda; ideal ≥ 5.000 fotos (aceitamos o que o cliente liberar).
- **Conteúdo:** o medidor **no contexto** (fachada, portão, caixa de padrão, instalação
  externa) — não só close do display — para treinar classificação de cena.
- **Rótulo mínimo por foto:**
  - código da ocorrência/nota aplicada pelo leiturista (I100, I300, …);
  - leitura registrada (valor do display, quando houver);
  - (se existir) decisão da fiscalização: **foto aceita/rejeitada** — isso é o rótulo
    ouro para treino supervisionado de detecção de fraude/erro.
- **Metadados valiosos (se disponíveis):** bairro/CEP ou região (para cruzar com a
  camada de perdas e priorizar fiscalização), tipo de medidor, data.
- **Formato:** imagens (jpg/png) + tabela CSV com os metadados.

### 2. Catálogo de ocorrências (leve)

Lista dos códigos de nota/ocorrência que a distribuidora usa (I100, I300, …) com o
**padrão esperado de foto** para cada um — vira a taxonomia de classes do modelo.

### 3. (Opcional, Camada 1) Benchmark real BR

Amostra de fotos de medidores **brasileiros da base da distribuidora** com a leitura
correta, para validar o OCR do MAPEN em condições reais de campo (o UFPR-AMR é a base
mais próxima, mas é antigo).

## Questões a levar

1. É possível disponibilizar esse lote de fotos de campo com as ocorrências? Qual o
   volume aproximado e o prazo?
2. A distribuidora tem **histórico auditado** (fotos que já passaram por revisão com
   decisão aceita/rejeitada)? Isso habilitaria treino supervisionado direto.
3. Quais tipos de ocorrência existem e existe um documento de padrões esperados de foto?
4. Como fica a **anonimização/LGPD**: fotos podem conter dados pessoais (fachada,
   placa, pessoas). Qual o fluxo de anonimização/termos de uso para pesquisa acadêmica?
5. A área tem métricas atuais de erro de leitura (ex.: % de fotos rejeitadas, taxas de
   ocorrência por tipo)? Servem de baseline de avaliação.
6. Dá para cruzar as fotos com a região (bairro/CEP) para priorizar fiscalização onde a
   perda é alta?

## Questões operacionais/técnicas (dos colegas — `perguntas_cliente_colegas.docx`)

Complementam as questões acima com detalhe operacional fino, presumindo acesso a artefatos reais
(`BaseExtracao_"data"_Dia.csv`, pastas de imagem, planilha de descrição de notas):

**Fluxo de conferência**
1. Como as imagens são baixadas para a máquina local? Com que frequência? Em lotes — de que tamanho?
2. O `BaseExtracao_"data"_Dia.csv` é a lista/planilha de controle que guia o trabalho do analista?
   Das 4 colunas (Número do medidor, Posição do medidor lida, Nota de Leitura Atual, Foto do
   medidor), quais vêm pré-preenchidas, em branco, ou precisam de conferência via foto?
3. O que faz uma foto ser rejeitada (ilegível, fora de contexto)? O que acontece na planilha de
   controle quando isso ocorre?
4. Após conferência, como a informação é consolidada no SAP? Sobe um CSV — é o mesmo
   `BaseExtracao_"data"_Dia.csv` ou outro template?

**Conteúdo e consistência dos dados**
5. Por que algumas linhas de "Número do medidor" têm mais de um código? Foi trocado o medidor?
   O número que aparece na foto sempre corresponde ao valor mais à esquerda na célula?
6. Por que existem fotos com "Nota de Leitura Atual" = NA? O que o analista deve conferir nesse caso?
7. Quando a nota exige foto mas "Foto do medidor" está NA, por que não há foto e o que acontece?
8. Na planilha "DESCRIÇÃO NOTAS LEITURISTAS X SOLICITAÇÃO DE FOTO", para quais códigos (NOTA) é
   necessário verificar o consumo ("Posição do medidor lida") a partir da imagem?
9. Por que há mais arquivos de imagem na pasta do que linhas listadas no CSV correspondente?

**Metadados e nomenclatura**
10. Qual a regra de nomenclatura das pastas locais (ex.: `PSP_EXTRATLEITIMPL_030726_0121`) e dos
    arquivos de imagem (ex.: `..._00000000000211765824_000`) — o que cada segmento significa?

**Outras**
11. Nome correto do departamento/cargo envolvido na conferência de imagens?
12. Qual o modelo do leitor PDA usado em campo?

## Plano B (se não houver lote real)

1. **Sintético:** gerar variações de foto de medidor + fachada (aumentação com dados
   abertos do MAPEN) com rótulos sintéticos de ocorrência.
2. **Piloto manual:** montar um mini-lote com poucas dezenas de fotos rotuladas à mão
   para provar o conceito e calibrar o pipeline.
3. Apresentar no Kickoff como "o que dá para fazer já" e o que fica bloqueado sem dados
   reais.
