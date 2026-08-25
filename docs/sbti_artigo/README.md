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

## Próximos passos

1. Decidir foco do artigo: (a) só Tarefa 1 (OCR de medidor, dados + benchmark prontos, resultado
   fechado) vs. (b) as duas tarefas incluindo o framing do problema de negócio (mais amplo, mas
   Tarefa 2 ainda é gap/proposta, não resultado).
2. Mapear seções do artigo real → template (Introdução, Trabalhos Relacionados/Contexto,
   Metodologia, Resultados, Conclusão) respeitando os nomes de seção do SBTI.
3. Escrever em Português (regra global de inglês não se aplica — artigo em veículo PT vai em PT,
   é conteúdo, não system prompt).
4. Redigir preservando anonimato (sem nomes de autor/instituição no corpo do PDF submetido).
