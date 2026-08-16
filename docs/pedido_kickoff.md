# Pedido de dados e questões para o Kickoff 12/09 — distribuidora

**Data:** 2026-08-08 | **Status:** rascunho pré-kickoff (validar com o professor/grupo)
**Referência:** `projeto4_desafio.md` (desafio), `analise_suficiencia.md` (gap)
**Meta:** sair do Kickoff com acesso a um lote real de fotos de campo da distribuidora.

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

## Plano B (se não houver lote real)

1. **Sintético:** gerar variações de foto de medidor + fachada (aumentação com dados
   abertos do MAPEN) com rótulos sintéticos de ocorrência.
2. **Piloto manual:** montar um mini-lote com poucas dezenas de fotos rotuladas à mão
   para provar o conceito e calibrar o pipeline.
3. Apresentar no Kickoff como "o que dá para fazer já" e o que fica bloqueado sem dados
   reais.
