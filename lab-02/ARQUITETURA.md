# Ficha de arquitetura — Luminus (Projeto 4, Lab 2)

## 1. Decisão apoiada
- **Decisão:** para cada registro de leitura do cliente, dar a cor **verde / amarelo / vermelho** que diz ao analista o que fazer
  com a foto. Verde: leitura da foto confirmada contra a digitada, sem olhar. Amarelo: o analista confere. Vermelho: foto
  inválida (sem medidor, completamente ilegível, ou nota que exige foto e a foto falta).
- **Unidade:** a foto de um registro (12.340 JPGs, ~13,7 mil registros nos 4 lotes; volume adotado 50 a 70 mil fotos por mês).
- **Quem age:** o analista de conferência. **Erro mais caro:** o **falso-verde** (fatura errada). Meta: amarelo < 40% do volume.
- **Onde roda:** CPU, em lote, retomável (CLI + BentoML opcional). Sem GPU presumida.

## 2. Sub-tarefas
| Sub-tarefa | Aprendida ou regra? | Tipo | Saída | Ativação | Perda | Rótulo |
|---|---|---|---|---|---|---|
| Há medidor na foto? | aprendida | classificação binária | 1 logit | sigmoid | BCE ponderada | fraco (LLM), 3.247 fotos |
| O display é legível? | aprendida | classificação binária | 1 logit | sigmoid | BCE ponderada | fraco (LLM) → C2 humano |
| Onde está o visor (caixa)? | aprendida | segmentação (mapa DB) | mapa 1×H×W | sigmoid | BCE (hard negatives) + dice | quads do visor, 1.091 |
| Qual caixa é o visor? | aprendida | classificação binária | 1 logit | sigmoid | BCE | candidatos por IoU, 27 mil |
| **Ler os dígitos (bloco profundo)** | **aprendida** | **sequência (CTC)** | T×11 | softmax por passo | CTC | leitura digitada confirmada / C2 |
| Leitura == digitada | **regra** | comparação | verde/não | — | — | — |
| Nitidez (Laplaciano) | **regra** | limiar | legível/não | — | — | — |
| Nota exige foto? (catálogo, NA) | **regra** | tabela do cliente | R1/R1b/R2/R2b | — | — | planilha do cliente |
| Cor final (R1…R6) | **regra** | cascata de regras | cor + motivo nomeado | — | — | — |

Comparar o valor lido com o digitado e decidir a cor **não** é tarefa de rede: é regra auditável com motivo nomeado.

## 3. Pipeline
```
foto ─▶ rotação ─▶ det do visor (PP-OCRv5 orig. + fine-tunado) ─▶ qual caixa é o visor? ─▶ CRNNDigitos ─▶ regras R1..R6 ─▶ cor
          └──▶ "display legível?" ─────────────────────────────────────────────────────────────▶ R6: ilegível ⇒ vermelho
```
Taxas medidas (amostra de 454 fotos sem vazamento; ver `docs/2026-10-08_cena_det_e_retest_amarelo.md`):

| Estágio | Taxa | Origem |
|---|---|---|
| Visor entre os candidatos (recall do det, fotos em que já havia candidato) | 0,84 | medida (valid+test, IoU ≥ 0,5) |
| Candidato certo escolhido | 0,98 | medida (top-1, test; regra antiga 0,46) |
| Leitura exata no recorte | 0,52 | medida (lote de teste) |
| Display legível identificado | F1 0,79 | medida (rótulo fraco) |

Produto ponta a ponta medido (não estimado): **verde 27,5%, amarelo 67,7% (43,1% com R6 ilegível→vermelho), vermelho 4,8% (29,4% com R6)**,
com `NA` sem foto = verde. O erro se multiplica: 0,84 × 0,98 × 0,52 ≈ 0,43 de fotos com leitura correta, que a regra de coerência
confirma em 27,5% (as demais ficam amarelas por motivo nomeado).

## 4. Contrato de entrada (por modelo)
| Modelo | Tamanho | Proporção | Normalização | Canais |
|---|---|---|---|---|
| Det do visor | lado maior ≤ 960, padding múltiplo de 32 | preservada | ImageNet (média/desvio) | RGB |
| Cena / "display legível?" | 320×240 | foto em pé 3:4 | ImageNet | RGB |
| "Qual caixa é o visor?" | 64×192 | recorte ~3:1 | ImageNet | RGB |
| **CRNNDigitos** | **32×128** (`--width` configurável) | squash (display ~5:1) | [0, 1] | cinza |

## 5. Espinha dorsal
- **Leitor:** CNN de 4 blocos + GRU bi-direcional (**0,44 M parâmetros**). Descartados: **TrOCR-small** (≈62 M, e é 93% da
  latência: média 3,7 s, p90 10,8 s; inventa sequências: 83 de 138 erros), **PP-OCRv6 tiny** (1,1 M; leitura exata 0,26 no
  cliente contra 0,52 do nosso ajustado, e não é nosso para ajustar com CTC curta).
- **Classificadores:** MobileNetV3-Small (≈1,5 M com cabeça de 1 logit) em vez de ResNet-18 (11,7 M): 7× menos parâmetros e a
  implantação é CPU.
- **Detector:** PP-OCRv5 mobile (1,2 M) em vez de YOLO/Ultralytics (**AGPL-3.0**: consequência para produto entregue ao cliente) e
  de SSDLite (BSD, mas exigiria treinar do zero o que o PP-OCR já resolve parcialmente).
- **Normalização:** BatchNorm em vez de GroupNorm: medido, GroupNorm não sobreajusta nem um lote de 32 recortes (0,00 × 1,00).

## 6. Cabeças
| Cabeça | Nº saídas | Ativação | Perda | Classes exclusivas? Por quê |
|---|---|---|---|---|
| CRNN | 11 por passo (T = 32) | softmax (dentro da perda) | CTC, blank = 0 | **por passo sim** (um símbolo por passo); por sequência não (vários dígitos). 795/2.547 leituras reais repetem dígito, o branco evita colapsar `99` em `9` |
| Há medidor / legível | 1 | sigmoid | BCE com logits | binárias; "legível" e "sem medidor" não são independentes (ilegível inclui sem medidor) |
| Visor (det) | mapa | sigmoid | BCE + dice | por pixel, 1 classe |

## 7. Desbalanceamento
Medido: "sem medidor" 8% (269/3.247); "ilegível" 38% (1.248/3.247); candidatos positivos do visor 10,8% (2.318/21.550); comprimento
da leitura 54,9% com 5 dígitos, 39,6% com 4 (UFPR-AMR). Técnica na perda: `pos_weight` na BCE das binárias; reamostragem 1:3
positivo:negativo por época no classificador de visor; comprimentos sintéticos amostrados da distribuição real.

## 8. Capacidade × rótulos
547 recortes de treino do cliente (lotes de treino) + 1.400 do UFPR-AMR; sem rótulo humano ainda (C2 pendente). Estratégia:
**transferência na mesma tarefa**: CRNN treinado no UFPR-AMR (0,873) e **ajuste fino sem congelar** no cliente (D3: congelar a CNN dá
0,376 contra 0,521). Parâmetros: 441.931 totais, 441.931 treináveis (ajuste fino) ou 200.971 (extração).

## 9. Confiança e recusa
Confiança = média geométrica das probabilidades máximas por passo da CTC; **temperatura T = 3,76 ajustada no valid** (ECE no teste
0,432 → 0,193). Regra de decisão: verde só se leitura == digitada **e** confiança ≥ limiar escolhido no valid **e** foto legível.
**Meta "cobertura ≥ 60% com risco ≤ 2%" não cabe na curva** (risco 26,8% a 60% de cobertura; cobertura 0,4% a 2% de risco): ver README §E.

## 10. Aumento de dados
| Transformação | Modelo | Muda o rótulo? |
|---|---|---|
| Brilho/contraste, giro ±10°, desfoque, jitter de recorte ±7% | CRNN, visor, det | não |
| Inversão de polaridade (p = 0,4; 41,7% dos recortes reais) | CRNN, det | não |
| Glare / oclusão parcial / ruído **por dígito antes de concatenar** | CRNN (sintético) | não |
| Espelhamento, rotação grande, deformação elástica | — | **descartadas** (dígito espelhado não existe; distorce segmentos) |
Currículo: as 6 primeiras épocas sem augmentation (a CTC trava no platô de brancos com augmentation desde o início).

## 11. Partição
Por **lote**: treino `200526_0352` + `210526_0335` (547), valid `220526_0408` (264), teste `030726_0121` (280). Interseção de
medidores, fotos e imagens entre conjuntos: **0** em todos os pares (`leiturista split-lote`; saída no notebook). Limiares e temperatura
escolhidos só no valid.

## 12. Sanidade
Formas: `1×32×128 → … → 32×256 → 32×11` (B1). Perda inicial (D1): 14,3 contra 17,05 da saída uniforme. Sobreajuste de 16 recortes reais
(D2): 14,33 → 0,004, 16/16 exatas. Baseline (D3): HOG + linear = 0,000; CRNN extração 0,376 ± 0,011; ajuste fino 0,521. Saídas completas
em `lab02.ipynb`.

## 13. Implantação
Volume: 50 a 70 mil fotos/mês. **Latência (60 fotos, CPU): p50 1,67 s, p90 11,2 s no pipeline atual**, dos quais **TrOCR é 93%**;
detecção 0,27 s, rotação 0,17 s. Com o CRNN (3,3 ms/recorte) no lugar do TrOCR, estimativa de ~0,5 a 0,7 s/foto. Exportação ONNX do
detector mantém o contrato do pipeline. Licenças: PP-OCR (Apache-2.0), torchvision (BSD), PyTorch (BSD); evitado YOLO/Ultralytics (AGPL-3.0).

## 14. O que decidimos não fazer
- **YOLO/Ultralytics**: licença AGPL-3.0 para produto entregue ao cliente.
- **Rede ponta a ponta** (foto → cor): não existe rótulo final, e a decisão é regra auditável.
- **TrOCR como leitor**: 93% da latência, 62 M parâmetros, inventa sequências.
- **GroupNorm**: medido pior (não sobreajusta um lote).
- **Pré-treino sintético** (LCD digits, 7 segmentos): não melhorou o UFPR-AMR (McNemar p = 0,50 a 1,00) e sozinho não generaliza (0,010); mantido só como augmentation.
- **Mais fine-tune do detector** sem rótulos manuais de caixa: o gargalo passou de "achar" para "escolher e ler".
