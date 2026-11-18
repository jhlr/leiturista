# Monitor visual, aceleração por GPU (DirectML) e otimização da heurística de leitura e serial

**Data:** 2026-11-17 | **Status:** implementado e validado visualmente via dashboard | **Escopo:** monitoramento batch, latência e extração de serial/display.

---

## 1. Motivação e Contexto

Com o pipeline de inferência estruturado e os modelos treinados/fine-tunados em releases anteriores, surgiu a necessidade de:
1. **Acompanhamento visual em batch:** monitorar a ferramenta em tempo real, verificando a distribuição da triagem semáforo (Verde / Amarelo / Vermelho), métricas por foto e debug de caixas detectadas.
2. **Redução de latência:** a inferência estava demorando vários segundos por foto.
3. **Correção da extração do serial:** nas fotos de campo, o serial do medidor não era identificado ou vinha corrompido, e frequentemente roubava o lugar da leitura do visor.

---

## 2. App de Monitoramento Visual (`app/monitor.py`)

Desenvolvido dashboard interativo em Streamlit (`.venv/Scripts/streamlit.exe run app/monitor.py --server.port 8502`) contendo:

- **Resumo Semáforo:** contagem e percentual de fotos em cada bucket (Verde: auto-aprovação / Amarelo: revisão humana / Vermelho: rejeição/sem leitura) com barra proporcional visual.
- **Instrumentação de Latência por Estágio:** medição transparente de tempo decorrido em cada etapa (`det`, `trocr`, `ppocr`, `rotacao` e pós-processamento), expondo média, fração percentual e gráfico comparativo.
- **Galeria Interativa:** navegação visual pelas imagens anotadas (bounding boxes coloridas) filtradas por bucket.
- **Inspeção Individual com Painel de Debug:** visualização individual em alta resolução, flags levantadas e tabela completa de todas as caixas de texto brutas reconhecidas pelo OCR com suas confianças e campos atribuídos.
- **Exportação CSV:** geração e download de relatório completo de auditoria do lote processado.

---

## 3. Aceleração por GPU (DirectML no Windows)

O ambiente de execução possui GPU integrada **AMD Radeon (TM) Graphics**. Como não há suporte CUDA (específico para NVIDIA), a aceleração de hardware foi habilitada via **DirectML**:

- **Modelos ONNX (`src/leiturista/inference.py`):**
  Detecção (`PP-OCRv5`) e reconhecimento leve (`PP-OCRv6_tiny`) foram configurados para carregar preferencialmente `DmlExecutionProvider` (com fallback automático para `CPUExecutionProvider`):
  ```python
  providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
  self._det = ort.InferenceSession(str(DET_ONNX), providers=providers)
  self._rec = ort.InferenceSession(str(REC_ONNX), providers=providers)
  ```
- **Dispositivo Dinâmico (`src/leiturista/models.py`):**
  Ajustado `resolve_device()` e `_device_to_torch()` para suporte a DirectML com fallback robusto para CPU em ambientes com Python 3.14.

---

## 4. Otimização de Latência do Pipeline

Mesmo com a GPU ativa, observou-se lentidão residual decorrente da ordem de execução no reconhecimento:

1. **Inversão da Prioridade no Reconhecimento de Recortes:**
   - **Antes:** `text = self._trocr_recognize(crop) or self._rec_recognize(crop)` executava o TrOCR (modelo autoregressivo de 62M parâmetros em CPU) em **cada recorte de texto** da imagem, deixando o PP-OCRv6 na GPU como mero fallback.
   - **Depois:** invertido para priorizar o PP-OCRv6 acelerado na GPU (`text = self._rec_recognize(crop) or self._trocr_recognize(crop)`), acionando o TrOCR apenas se o PP-OCR retornar vazio.
2. **Greedy Search no Fallback do TrOCR:**
   - Reduzido `num_beams=4` para `num_beams=1` em `_trocr_recognize()`, tornando qualquer invocação residual de fallback até 4x mais rápida em CPU.
3. **Resultado:**
   - A latência por foto caiu de vários segundos para a faixa de ~1,5s a 2s (incluindo rotação quádrupla e detecção normal + invertida).

---

## 5. Diagnóstico e Correção da Heurística de Serial e Leitura

### 5.1 Diagnóstico do Erro Anterior
Nas fotos da distribuidora parceira, o número de série do medidor fica impresso na placa (geralmente acima do código de barras) e possui formato **estritamente numérico** (7 a 11 dígitos). A lógica anterior causava duas falhas graves:
1. `_classify()` exigia pelo menos uma letra (`[A-Za-z]`) para classificar como `serial`. Consequentemente, o serial numérico nunca era classificado como serial.
2. A regra de desempate `_best_reading()` escolhia o candidato numérico com **mais dígitos**. Como o serial tem 7 a 10 dígitos e o display de kWh tem 4 a 6 dígitos, o **serial roubava o lugar do display** (virava a leitura de consumo), enquanto textos com letras na carcaça (ex: `220V`, `INMETRO`, modelo) eram indevidamente atribuídos como serial.
3. Caracteres confusos lidos pelo OCR como letras (ex: `'B'` no lugar de `'8'`) eram descartados por `_digits()`, fazendo dígitos sumirem (ex.: `B88012` virava `88012`, perdendo o primeiro número).

### 5.2 Correções Implementadas em `inference.py`

1. **Classificação Robusta de Serial Numérico e Display:**
   - Sequências de **7 a 14 dígitos** predominantemente numéricas agora são categorizadas como `serial`.
   - Displays de consumo são restritos à faixa padrão de **3 a 6 dígitos** (`leitura`).
   - Termos e especificações elétricas (`220V`, `60Hz`, `15(100)A`, `BIFASICO`, `TRIFASICO`, etc.) são filtrados via regex (`ELEC_SPEC_RE`) e descartados como ruído (`outro`).
2. **Desempate Espacial e Semântico:**
   - O visor de leitura é priorizado no terço superior (menor `y`) e na faixa ideal de 4 a 6 dígitos.
   - O serial é buscado entre caixas que **não sejam a leitura**, priorizando as caixas localizadas na placa/código de barras (maior `y`).
3. **Normalizador de Confusões de OCR (`_ocr_to_digits`):**
   - Substituição antes do descarte: `B/&` $\to$ `8`, `O/o/D` $\to$ `0`, `S/s/$` $\to$ `5`, `I/i/l/|` $\to$ `1`, `Z/z` $\to$ `2`, `A` $\to$ `4`, `b/G` $\to$ `6`.
   - Recupera dígitos que seriam excluídos do display ou serial.
4. **Limpeza e Formatação do Serial:**
   - Remoção de prefixos (`SN:`, `S/N`, `Nº`, `NR`, `MEDIDOR:`).
   - Eliminação de pontuações (`.`, `-`, `/`, `:`, espaços), entregando o serial 100% numérico.
5. **Margem de Recorte Aumentada:**
   - `pad` aumentado de `8` para `12` pixels e `DET_UNCLIP` de `1.5` para `1.6`, evitando corte de dígitos nas bordas.
6. **Ativação Padrão dos Modelos Especializados:**
   - `DET2_ONNX` (`models/pp_ocr_v5_mobile_det_visor_onnx/inference.onnx`) e `VISOR_CLS` (`models/visor_cls.pt`) passam a ser ativados automaticamente por padrão quando presentes no diretório `models/`.

---

## 6. Resultados Validados nas Amostras

| Imagem de Teste | Leitura Anterior | Leitura Nova | Serial Anterior | Serial Novo |
|---|---|---|---|---|
| `exemplo_01.png` | `0178655` (serial roubando leitura) | `017355` (visor correto) | `None` | `4138190` (numérico limpo) |
| `exemplo_02.png` | `201821` | `204927` | `None` | `None` |
| `exemplo_03.png` | `None` | `00234` | `5N21668732` (letras e prefixo) | `21668732` (serial numérico) |
| `exemplo_04.png` | `88012` (perdia 1º dígito) | `883012` (6 dígitos) | `None` | `70744075` |

---

## 7. Coerência Foto x Ocorrência de Campo (Tarefa 2)

### 7.1 A Foto "Ruim" como Sucesso da Evidência
Na operação da distribuidora, quando o leiturista não consegue obter a leitura por impedimento no imóvel (ex: casa fechada `I100`, medidor retirado `B100`, local demolido `D100`, cão bravo `I160`, display apagado `G100`), ele registra a ocorrência e fotografa o portão, a fachada ou a caixa vazia.
- **Antes:** O pipeline tratava a ausência de leitura nessas fotos como falha do modelo (classificando como 🔴 Vermelho / erro).
- **Depois:** O pipeline e o monitor compreendem que para ocorrências de impedimento, **a foto é feita para NÃO ter medidor nem leitura**. A ausência de leitura é a confirmação do impedimento relatado $\to$ **🟢 Verde (Evidência Consistente)**!

### 7.2 Módulo de Ocorrências (`src/leiturista/ocorrencias.py`)
- Catálogo completo de 61 notas da distribuidora, com descrição e obrigatoriedade de foto.
- Separação entre notas de `IMPEDIMENTO_AUSENCIA` e `LEITURA_REGULAR`.
- Função `avaliar_coerencia(pred, nota, leitura_registrada)` que determina o bucket semáforo com justificativa textual para o analista.

### 7.3 Integração no Monitor (`app/monitor.py`)
- O dashboard agora lê e indexa automaticamente os arquivos `BaseExtracao_*.csv` da pasta de fotos.
- Exibe colunas de Nota, Descrição da Ocorrência, Leitura do Sistema e Justificativa de Coerência.
- Inspeção individual detalha o motivo pelo qual a evidência foi aprovada como consistente ou direcionada para revisão.

---

## 8. Próximos Passos (Backlog)

1. Ajustar o pós-processamento para medidores com displays mecânicos/ciclométricos (odômetro).
2. Avaliar fine-tune de OCR específico para dígitos de medidor (ou uso do CRNN já treinado em `models/crnn_dist_v2_ft.pt`).
3. Integrar feedback de conferência humana diretamente do dashboard para fechamento do loop ativo.
