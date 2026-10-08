# Uso de IA generativa neste laboratório

Grupo Luminus (Projeto 4). Ferramenta: Claude Code (Anthropic), em sessão local no repositório, com leitura e escrita de
arquivos e execução de comandos. Método: **desenvolvimento orientado por especificação**. Para cada item, o integrante
escreveu a especificação (objetivo, restrições, critério de aceite), a ferramenta propôs a implementação, e o integrante
verificou contra o critério antes de aceitar. Nenhum resultado numérico foi aceito sem ser reproduzido por comando no
repositório (`leiturista ...`, `lab02.ipynb`), e todo número do `README.md` aponta para o arquivo que o gerou.

Os prompts abaixo estão resumidos pela intenção e pelas restrições; não são transcrição literal.

## 1. Registro por tarefa

### 1.1 Ficha de arquitetura (item A) — `ARQUITETURA.md`
- **Especificação:** preencher as 14 seções do guia; toda escolha com alternativa descartada e motivo; seção 2 separando o que é
  aprendido do que é regra; seção 3 com a conta do produto das taxas e a origem da estimativa.
- **Aceito:** estrutura e redação das seções; conferidos pelo grupo contra o pipeline real.
- **Descartado / corrigido:** versão inicial citava números que não estavam em nenhum arquivo do repositório; foram removidos
  ou trocados por medições em `docs/*.json`.
- **Verificação:** cada número da ficha tem fonte em `docs/` ou na saída do notebook.

### 1.2 Modelo em código (itens B1–B3) — `CRNNDigitos`
- **Especificação:** CNN + GRU bidirecional + CTC; entrada 1×32×128; tabela de formas; parâmetros totais e treináveis nas
  configurações de extração e ajuste fino; justificar ativação e perda com exemplo real da base.
- **Aceito:** a arquitetura e a contagem de parâmetros (441.931 totais; 200.971 treináveis em extração).
- **Descartado:** o `main()` do esqueleto do docente chama `tabela_de_formas` e `sobreajustar_um_lote` como métodos da
  instância, mas são funções soltas (`AttributeError` no bloco do CRNN); a chamada foi corrigida no notebook.
- **Verificação:** o exemplo de dígito repetido (795 de 2.547 leituras reais) foi contado por consulta na base, não estimado.

### 1.3 Partição por lote (item C4)
- **Especificação:** treino, validação e teste separados por lote; demonstrar interseção zero de medidor, foto e imagem.
- **Aceito:** `split_by_lote` e a saída no notebook (0 em todos os pares).
- **Descartado:** partição aleatória por foto, que dava 0,609 de leitura exata e foi substituída por lote (0,521). O número
  menor é o que está reportado.

### 1.4 Sanidade, baseline e extração × ajuste fino (itens D1–D3)
- **Especificação:** perda inicial comparada a T·ln(11)/L; sobreajuste de 16 recortes reais; 3 sementes, mesma partição,
  mesma métrica; reportar se o profundo perder.
- **Aceito:** D1 e D2 como medidos; a baseline HOG + regressão logística com 0,000.
- **Descartado:** a comparação direta de 14,3 com ln(11) (a CTC divide pelo comprimento do alvo, a referência correta é
  32·ln(11)/L̄ = 17,05). A baseline ficou registrada com a ressalva de que uma baseline mais forte (segmentação + SVM) não foi
  tentada.
- **Correção do integrante:** a ideia inicial de congelar a CNN no ajuste fino foi rejeitada; só faz sentido para espinha
  genérica (ImageNet), e aqui a CNN já foi treinada na mesma tarefa. O D3 mediu o custo: 0,376 congelada contra 0,521.

### 1.5 Calibração e cobertura × risco (E1–E2)
- **Especificação:** temperatura ajustada **só na validação**; ECE antes e depois; curva cobertura × risco; marcar a meta do
  kickoff e dizer se não cabe.
- **Aceito:** T = 3,76, ECE 0,432 → 0,193; a conclusão de que a meta não cabe na curva (cobertura 60% com risco 26,8%).
- **Verificação:** temperatura ajustada em 264 recortes da validação; teste usado apenas para reportar.

### 1.6 Parte específica Luminus (L1–L4)
- **Especificação:** classificar erros (perdeu, trocou, duplicou, inventou) com critério reproduzível; benchmark no cliente
  contra o UFPR-AMR; métrica otimizadora única com restrições; medir latência por estágio.
- **Aceito:** `leiturista.errors.classify_error` (alinhamento de edição); medição de latência em 60 fotos, CPU.
- **Descartado:** a hipótese de que o gargalo do p90 estaria na detecção ou na rotação; a medição mostrou o TrOCR com 93% do
  tempo.
- **Achado registrado:** ligar toda a augmentation desde a época 0 travou a CTC emitindo só branco; resolvido com currículo.

### 1.7 Plano de treino em três estágios (pré-treino sintético → ajuste fino)
- **Especificação:** documento único em `docs/dl-lab2/plano-treino-crnn.md`, com rastreabilidade para o enunciado.
- **Revisão por segundo agente sem o contexto da sessão**, com a persona de professor de MLOps/DL/CV. Cinco correções
  incorporadas: (a) vazamento de informação entre configurações (fixar hiperparâmetros antes de olhar o teste da configuração
  anterior); (b) teste de McNemar pareado em vez de comparar a olho com 300 exemplos; (c) medir o orçamento de tempo do
  estágio 1 antes de rodar; (d) critério reproduzível de clipping (percentil 90 da norma do gradiente); (e) anotar o risco da
  CTC com sequência sintética de comprimento 1–2.
- **Responsabilidade:** o treino foi disparado e avaliado pelo integrante; a ferramenta preparou a pipeline, não executou
  `leiturista train`.

### 1.8 Rotulagem (itens C1-C3)
- **Especificação:** esquema de classes com exemplos-limite; plano de 300 fotos estratificadas por lote com 50 em dupla; app de
  rotulagem; cálculo de kappa por classe.
- **Aceito:** `rotulos/plano.csv`, `app/rotular.py`, `leiturista rotulos-kappa`, e depois a conversão das fontes reais de rótulo
  para o esquema (`rotulos/rotulos.csv`, `esquema.md` revisado).
- **O que ocorreu de fato:** o plano de 300 fotos com dupla humana não foi cumprido por falta de tempo. Entraram duas fontes:
  triagem humana do Mikael (3.063 fotos, um lote, em 3 pastas) e Llama-3.2-Vision 11B local (3.247 fotos), ambos rotulando
  sem ver o rótulo do outro. O C3 foi calculado como humano × modelo (688 fotos, kappa 0,324), declarado assim no README.
- **Uso da ferramenta de IA aqui:** conversão das pastas e das colunas `llm_*` para o esquema, cálculo do kappa com a função do
  repositório, auditoria visual de 12 fotos. **Não rotulou nenhuma foto no lugar de humano.** A auditoria visual inicial por
  miniaturas gerou uma hipótese que não se sustentou; o `esquema.md` registra a correção.
- **Descartado:** a hipótese de que "medidoruim" fosse "medidor em mau estado" (o autor confirmou: significa ilegível).

## 2. Limites que o grupo impôs ao uso da ferramenta

1. Treino e avaliação final de modelo são disparados por um integrante.
2. Nenhuma foto do cliente foi enviada a serviço externo; as fotos não estão no repositório. Os notebooks são limpos de
   saídas antes do commit, para que não carreguem imagem do cliente.
3. Número sem fonte reproduzível não entra no relatório; resultado negativo entra como medido (baseline em 0,000; meta de
   cobertura × risco que não cabe).
4. Código gerado só entra depois de executado; testes de guarda só entram depois de vistos falhar contra o código sem a
   correção.

## 3. Registro individual

| Integrante | Ferramenta | O que pedi | O que aceitei | O que descartei e por quê |
|---|---|---|---|---|
| João Rietra (jhlr) | Claude Code | Ficha, tabela de formas, D1-D3, E1-E2, L1-L4, plano de treino, conversão de rótulos, revisão do esquema, conferência do README | Estrutura da ficha, correção do esqueleto do docente, calibração só no valid, conversão das fontes de rótulo, as 5 correções da revisão independente | Congelar a CNN no ajuste fino (custou 14 pontos); comparação ingênua da perda inicial com ln(11); hipótese "sequência incompleta" e hipótese "medidor em mau estado" para o desacordo de rótulos (refutadas pela amostra e pelo autor da triagem) |

Rotulagem humana (triagem em 3 pastas): Mikael. Segundo rotulador: Llama-3.2-Vision 11B local.

## Declaração

O trabalho desta entrega foi feito por um único integrante, que é capaz de explicar qualquer linha do código entregue.
