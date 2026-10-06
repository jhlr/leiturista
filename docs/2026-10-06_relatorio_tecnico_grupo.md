# Relatório técnico para o grupo — LEITURISTA (entrega SR1)

Data: 2026-10-06. Autor do relatório: resumo do trabalho técnico feito até a entrega.
Complementa `docs/2026-10-06_resumo_tecnico_sr1.md` (respostas às 4 perguntas da apresentação).

## 1. Problema e escopo

Leitura automática de medidor de energia elétrica a partir de foto: extrair o número do
display (consumo), o serial da placa e sinalizar incoerências (foto ilegível, sem medidor,
mais de uma leitura). Duas tarefas: (1) leitura/OCR, com dados públicos suficientes
(UFPR-AMR); (2) validação foto ↔ ocorrência do leiturista, sem dataset público, dependente
do lote real da distribuidora. A entrega SR1 cobre só a camada de serviço (API), sem
front-end, treino novo ou deploy em nuvem.

## 2. Arquitetura

```
foto (multipart) -> POST /predict (BentoML) -> MeterOCR.predict_image
   det PP-OCRv5 (ONNX) -> caixas de texto
   rec PP-OCRv6_tiny (ONNX) -> lê cada crop
   classifica cada caixa: leitura | serial | outro
   flags de coerência (nitidez por variância do Laplaciano, múltiplas leituras, sem texto)
   fallback: sem display segmentado -> TrOCR-small-printed na imagem inteira
   tentativa extra com imagem invertida (display claro-em-escuro)
-> JSON {numero_medidor, funcao, consumo, confianca, legivel, flags}
```

Código em `src/leiturista/` (biblioteca instalável): `service.py` (endpoint, sem lógica
de OCR), `inference.py` (`MeterOCR`), `models.py`, `crnn.py`, `train.py`, `eval.py`,
`data.py`, `distribuidora.py`, `artifacts.py`, `cli.py`, `paths.py`.

Regra de `funcao`: `sem_leitura_detectada` (nada lido), `leitura_normal` (legível e
confiança >= 0,5), `leitura_incerta` (demais; vai para conferência manual).

## 3. Modelos e resultados

Nenhum modelo do pipeline em produção foi treinado pelo grupo.

| Papel | Modelo | Resultado (UFPR-AMR, 300 imgs de teste, display recortado) |
|---|---|---|
| Detecção | PP-OCRv5_mobile_det | — |
| Reconhecimento | PP-OCRv6_tiny_rec | exact-match 0,357; digit-acc 0,846 |
| Fallback | TrOCR-small-printed (off-the-shelf) | exact-match 0,253; digit-acc 0,776 |

Experimentos fora do serviço:
1. Fine-tune do TrOCR-small no UFPR-AMR (MLflow, run `9c14db62`). O checkpoint em uso no
   serviço continua sendo o off-the-shelf.
2. `CRNNDigitos` (CNN + GRU bidirecional + CTC, entrada 32x128 cinza, 10 dígitos + branco),
   treinado do zero em `data/finetune_ufpramr` (1.400/300/300). Receita: Adam lr 1e-3,
   lote 32, 15 épocas, seed 0. Valid: exato 0,910, dígito 0,967. Test: exato 0,873,
   dígito 0,945. Publicado como `crnn_digitos.pt` no release `modelos-1.1`. Treino só
   em recortes do UFPR-AMR; não validado em foto de campo.
3. Baseline de triagem verde/amarelo/vermelho (PP-OCR e CRNN) e `train-crnn` com vários
   datasets, `--init` (fine-tune) e augmentation de polaridade/brilho.

## 4. Limites conhecidos

1. **Localizar o display é o gargalo.** No lote de campo, só ~15% das fotos (1.087 de
   7.265) tiveram crop de leitura localizável no import calibrado. As heurísticas desse
   import não estão no `/predict`; foto de campo crua pode devolver lixo (ex.: placa
   lida como display).
2. Os números acima são em display já recortado, então não representam foto de campo.
3. `funcao` é heurística; não replica o catálogo de notas do cliente.
4. "Sem medidor" e "ilegível" caem no mesmo valor.
5. `confianca` é score do OCR, não probabilidade calibrada.
6. A Tarefa 2 (validação foto ↔ nota) não foi atacada: falta rótulo real de decisão.

Próximos passos propostos: detector de visor treinado em foto de campo; classificador
"tem medidor?" antes do OCR; fine-tune com dado de campo; mapear `funcao` para o catálogo
real; calibrar a confiança.

## 5. Reprodutibilidade e entrega

- `pyproject.toml` + `uv.lock`, `.python-version` 3.13, `justfile` (`setup`, `models`,
  `samples`, `serve`, `predict`, `all`), `.env.example`, `LICENSE` (MIT).
- Pesos fora do git: `just models` baixa `modelos-1.0` (147 MB) do release do GitHub.
- Imagens de `samples/` sintéticas (`scripts/gen_sample_images.py`, seed fixa), não versionadas.
- **Teste de clone limpo em 2026-10-06:** `git clone` do repo público, `just all` (exit 0:
  deps, 147 MB de pesos, 4 imagens) e `just serve` + `just predict`. Resposta para
  `exemplo_01.png`: consumo `017355`, `leitura_normal`, confiança 0,987, flag de imagem
  invertida e flag de serial não detectado (a imagem só tem display). HTTP 200, ~3,9 s na
  primeira chamada (carga do TrOCR).

## 6. Higiene do repositório (feita em 2026-10-06)

1. Dados reais do cliente (CSV de campo rotulado e 2 JSONLs de rótulos LLM) e o nome do
   cliente estavam no histórico. O histórico foi reescrito com `git filter-repo`, esses
   arquivos saíram de todos os commits e o nome virou "distribuidora".
2. Refs de PR do GitHub ainda guardavam commits antigos e não podem ser apagadas por
   push. O repositório foi apagado e recriado com o mesmo nome (público), reenviando
   `main` e as tags `sr1`, `modelos-1.0`, `modelos-1.1` (todas no commit limpo
   `d645587`), e os dois releases foram republicados com seus assets.
3. Consequência: PRs/issues antigos se perderam. Quem tem clone local antigo deve fazer
   `git fetch origin && git checkout main && git reset --hard origin/main` e **não** dar
   push do histórico antigo.
4. Os dados continuam só no disco local de quem os tem, cobertos pelo `.gitignore`.
   Regra do grupo: nunca commitar foto, CSV real ou nome do cliente.

## 7. Pendências

1. Vídeo de 2 min do serviço rodando (plano B da apresentação de 07/10).
2. Conferir se o professor precisa ser avisado sobre a exposição temporária dos dados.
3. Revisões entre integrantes (poucos PRs até agora).
4. Tarefa 2 (validação de coerência) depende do lote real da distribuidora.
