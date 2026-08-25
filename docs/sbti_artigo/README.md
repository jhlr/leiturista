# Artigo SBTI 2026 — planejamento

**Data:** 2026-08-25 | **Evento:** SBTI 2026 (Simpósio Brasileiro de Tecnologia da Informação, UFPE)

Fontes originais (baixadas em 2026-08-25/19, cópias comitadas nesta pasta):

- `entendimento_problema.docx` — enunciado do problema entregue pelo Grupo 2 (Luminus) à
  disciplina Projeto 4 (Cesar School, BD2026.2). **Atenção:** é o mesmo desafio de fiscalização
  fotográfica de leituristas, mas de OUTRO grupo/cliente (rótulos I100 etc.), não o `leiturista`
  deste repo diretamente — usar como referência de framing do problema de negócio, não como fonte
  de dados.
- `regras_submissao.pdf` — regras oficiais de submissão SBTI 2026.
- `template_edicao.docx` — template Word oficial do artigo.

## Regras de submissão — resumo operacional

| Item | Valor |
|---|---|
| Idioma | Português (aceita inglês) |
| Formato de envio | PDF, até 2 MB (versão final aceita: Word) |
| Página | A4, margens sup/esq 3 cm, inf/dir 2 cm |
| Fonte | Times New Roman 12 (citações e legendas de figura/tabela: 10) |
| Espaçamento | Simples entre linhas; 6 pt de salto após parágrafo |
| Extensão | 12–20 páginas (incluindo figuras, quadros, tabelas, referências) |
| Título | Times New Roman 16, negrito, centralizado, até ~30 palavras |
| Resumo/Abstract | 100–200 palavras, PT e EN, fonte 10 |
| Palavras-chave | 3–5, PT e EN, separadas por `;` |
| Referências | ABNT vigente |
| Anonimato | **Obrigatório** — sem nome de autor/instituição no corpo, resumo ou metadados ocultos do arquivo. Trabalho identificado é desclassificado. |
| Autoria | Máx. 3 artigos por autor; máx. 5 autores por artigo |
| Revisão | Double-blind |
| Publicação | Artigos aprovados no topo do ranking → periódico Qualis B1; demais aprovados → periódico B3 |
| Envio | `evento.sbti@ufpe.br` |

## Estrutura do template (seções nível 1)

1. INTRODUÇÃO
2. CONTEXTO (com subseção 2.1 Referências, estilo diferente do nível 1; nota sobre 2.2
   "Observações dos Avaliadores" é instrução do template, não seção do artigo real)
3. [seção de achados/resultados — o template usa "FINDANDO" como placeholder, adaptar título]
4. REFERÊNCIAS

Duas linhas em branco entre seções de nível 1. Parágrafos com `<tab>`. Texto sempre justificado.

## Ângulo do artigo — o que já temos para contar

Nossa solução real (`leiturista`, este repo) já tem material publicável:

- **Pipeline off-the-shelf funcionando** (det PP-OCRv5 + rec TrOCR/PP-OCRv6 + fallback) —
  `docs/relatorio_benchmark.md`.
- **Achados técnicos com payoff de artigo** (novidade metodológica, não só resultado):
  rotação-primeiro em vez de força bruta, duas fases de inferência (normal + invertida) para
  odômetro claro-em-escuro, seleção de leitura anti-gambiarra, assinatura do medidor via hash
  de tokens (`meter_similarity`) para re-identificação sem serial legível.
- **Benchmark quantificado** em dataset público (UFPR-AMR, Laroca IJCNN 2020): PP-OCRv6 35,7%
  exact-match / 84,6% digit-acc; TrOCR limpo 25,3% / 77,6%.
- **Gap identificado e documentado** para a Tarefa 2 (coerência foto↔ocorrência) —
  `docs/analise_suficiencia.md` — honesto sobre o que falta (dataset com notas do leiturista).

## Escopo decidido (2026-08-25)

**Tudo:** framing completo do problema de negócio (Tarefa 1 + Tarefa 2), os achados de engenharia
como método reproduzível, e o benchmark como validação quantitativa. Artigo não se limita a "só
OCR" nem só a heurísticas — conta a história completa: problema → pipeline → achados → números →
gap da Tarefa 2 como trabalho futuro honesto.

## Estrutura proposta (mapeada ao template SBTI)

1. **INTRODUÇÃO** — contexto do problema de negócio (fiscalização de fotos de leitura, volume
   50–70 mil imagens/mês, 6 analistas, carga de horas extras), motivação e objetivo geral.
   Fonte: `entendimento_problema.docx` (framing, generalizado — sem dados específicos do cliente
   de origem do docx) + `AGENTS.md` (objetivo do nosso projeto).
2. **CONTEXTO** (trabalhos relacionados) — UFPR-AMR/Laroca IJCNN 2020, PP-OCR (PaddleOCR), TrOCR
   (AAAI 2023); posicionar o gap de datasets públicos para Tarefa 2 (coerência foto↔ocorrência).
   - 2.1 Dados e métricas (dataset, split, exact-match, digit-acc).
3. **MÉTODO / PIPELINE** — arquitetura det→rec→classificação→seleção; os 6 achados de engenharia
   como contribuições método (rotação-primeiro, duas fases, retry invertido, seleção anti-gambiarra,
   serial vertical, assinatura do medidor via hash de tokens).
4. **RESULTADOS** — benchmark Tarefa 1 (PP-OCRv6 35,7%/84,6%; TrOCR limpo 25,3%/77,6%), achados de
   campo (Praekelt), status da Tarefa 2 como gap identificado e caminho proposto (piloto + dados
   reais no Kickoff).
5. **CONCLUSÃO** — ganho operacional esperado (redução de triagem manual), limitações honestas
   (serial heurístico, legibilidade por Laplacian, números off-the-shelf sem fine-tune),
   trabalhos futuros (fine-tune, dataset Tarefa 2).
6. **REFERÊNCIAS** — ABNT: Laroca et al. 2020, Li et al. (TrOCR) 2023, PaddleOCR/PP-OCR.

## Próximos passos

1. Redigir seção a seção em Português, respeitando anonimato (sem nome de autor/instituição no
   corpo nem em metadados do arquivo) e a formatação do template (TNR 12, A4, margens 3/2 cm,
   12–20 páginas).
2. Reaproveitar prosa de `docs/relatorio_benchmark.md` e `docs/analise_suficiencia.md` como base,
   reescrevendo em registro acadêmico (não copiar tabelas/markdown cru).
3. Gerar figura(s) do pipeline (diagrama do fluxo det→rec→seleção) para a seção de método.
4. Rodar a skill `academic-article-writing` para lapidar registro formal e citações ABNT antes da
   submissão.
