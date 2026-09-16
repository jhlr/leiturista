# Leiturista

Triagem automática de fotos de leitura de medidor de energia elétrica: a partir de
uma foto, o serviço extrai o número do medidor, classifica se a leitura é confiável
o bastante pra dispensar conferência manual, o valor lido e um grau de confiança.

Projeto da disciplina Projeto 4 / DADOS (Cesar School). O objeto original é uma
distribuidora de energia elétrica parceira da disciplina — o nome dela não aparece
neste repositório por acordo entre a instituição e a empresa; aqui ela é só
"a distribuidora".

**Escopo desta entrega (SR1):** camada de serviço — a inferência exposta atrás de
uma API que outra pessoa/programa consegue chamar. Sem front-end, sem integração
com sistemas do cliente, sem deploy em nuvem. Ver `docs/projeto4_desafio.md` para o
desafio completo do semestre.

## 1. Modelo — o que está sendo carregado

O pipeline (`src/leiturista/inference.py::MeterOCR`) não é um modelo único, é uma
composição de 3 modelos **públicos, pré-treinados** (nenhum foi treinado do zero
por este grupo):

| Papel | Modelo | Origem | Treinado por |
|---|---|---|---|
| Detecção de texto na foto | `PP-OCRv5_mobile_det` (ONNX) | `PaddlePaddle/PP-OCRv5_mobile_det_onnx` (HF) | PaddlePaddle/Baidu |
| Reconhecimento (1ª tentativa, classifica campo `leitura`/`serial`) | `PP-OCRv6_tiny_rec` (ONNX, 1,1M params) | `PaddlePaddle/PP-OCRv6_tiny_rec_onnx` (HF) | PaddlePaddle/Baidu |
| Reconhecimento (fallback, quando o det não segmenta o display) | `TrOCR-small-printed` (62M params, MIT) | Microsoft, off-the-shelf | Microsoft |

Nenhum dos três foi fine-tunado com dados desta entrega — são modelos prontos, o que
o próprio enunciado permite explicitamente ("nenhuma dessas opções vale mais que as
outras nesta nota"). O grupo tentou um fine-tune do TrOCR no dataset público
**UFPR-AMR** (Laroca et al., IJCNN 2020 — `docs/finetune_trocr_ufpramr.md`), mas o
checkpoint salvo em produção hoje (`.model_cache/trocr-small-printed`, MLflow run
`9c14db62`) é o **off-the-shelf**, não o fine-tunado — documentando aqui com
honestidade em vez de inflar a história do modelo.

## 2. Limites do modelo

Medidos no benchmark público UFPR-AMR (300 imagens de teste, `docs/relatorio_benchmark.md`):

| Modelo | exact-match | digit-acc |
|---|---|---|
| PP-OCRv6_tiny_rec (usado no pipeline) | 0.357 | 0.846 |
| TrOCR-small-printed off-the-shelf (fallback) | 0.253 (limpo) | 0.776 (limpo) |

Isso é em fotos de laboratório, já recortadas no display. Em fotos de campo reais
(lote piloto da distribuidora — não versionado, ver seção 6), dois problemas
adicionais aparecem e são maiores que o erro do leitor em si:

1. **Localização do display é o gargalo, não a leitura.** No import batch calibrado
   pro lote real (`docs/fotos_distribuidora_dados.md`), só 1.087 de 7.265 fotos
   (~15%) tiveram um crop de leitura localizável — o resto não achou o display na
   foto (medidor mecânico, ângulo, distância, reflexo). Esse import usa heurísticas
   extras (faixa de dígitos, zeros à esquerda, rejeição de texto de unidade) que
   **não estão** no endpoint genérico `/predict` deste serviço — chamar `/predict`
   direto numa foto de campo real, sem esse recorte prévio, tende a devolver lixo
   (ex.: confundir a placa de identificação do medidor com o display).
2. **O endpoint não distingue "não consegui ler" de "medidor com ocorrência"** (casa
   fechada, leitura informada pelo cliente etc. — os ~61 códigos de nota do cliente,
   `docs/fotos_distribuidora_dados.md`). O campo `funcao` da resposta é uma heurística
   simples (reading + legibilidade + confiança), não replica essas notas.

**Ajustes necessários** pra virar produto: (a) treinar/ajustar um localizador de
display específico pra foto de campo (o gargalo real, por achado próprio), (b)
terminar o fine-tune do TrOCR com dados de campo (autorizado pela disciplina, mas
"treino só o usuário dispara" — não rodado nesta entrega), (c) mapear `funcao` pro
catálogo de notas real do cliente em vez da heurística atual.

## 3. Como rodar (do clone à primeira predição)

Testado com **Python 3.14** + `uv`. Tempo aproximado: ~3 min (setup) + ~2 min
(download de 147 MB de modelos) numa rede razoável.

```bash
git clone https://github.com/jhlr/leiturista.git && cd leiturista
just setup      # uv sync — instala as dependências (uv.lock)
just models     # baixa + extrai os pesos (release público modelos-1.0, 147 MB)
just serve      # sobe o serviço em http://localhost:3000
```

Sem `just` instalado: `brew install just` (macOS) ou `cargo install just`. Os
comandos equivalentes sem `just`:

```bash
uv sync
curl -L https://github.com/jhlr/leiturista/releases/download/modelos-1.0/leiturista-models.tar.gz -o leiturista-models.tar.gz
tar -xzf leiturista-models.tar.gz
uv run bentoml serve leiturista.service:LeituristaService --port 3000
```

O primeiro `predict` demora alguns segundos a mais (carga lazy dos modelos); os
seguintes são rápidos (CPU, sem GPU necessária).

## 4. Contrato da API

**`POST /predict`** — multipart, campo `image` (arquivo de imagem).

```bash
curl -s -F image=@samples/exemplo_01.png http://localhost:3000/predict
```

Resposta:

```json
{
  "numero_medidor": null,
  "funcao": "leitura_normal",
  "consumo": "017355",
  "confianca": 0.987,
  "legivel": true,
  "flags": ["leitura via imagem invertida (display claro-em-escuro)"]
}
```

- `funcao`: `leitura_normal` (confiável, dispensaria conferência manual) |
  `leitura_incerta` (leitura extraída, mas com baixa confiança/legibilidade — manda
  pra conferência) | `sem_leitura_detectada` (não achou display na foto).
- `confianca`: 0.0–1.0, score do bloco de texto reconhecido como leitura. Fica `0.0`
  quando a leitura veio do fallback TrOCR na imagem inteira (sem box localizado).
- Swagger/OpenAPI interativo: `http://localhost:3000/` (a própria raiz do serviço,
  não `/docs` — BentoML 1.4 serve a UI ali).

Casos de teste (`samples/`, todos sintéticos — ver seção 6):

| Imagem | Resultado esperado |
|---|---|
| `exemplo_01.png` | leitura normal, alta confiança |
| `exemplo_03.png` | leitura normal + serial detectado |
| foto sem medidor (qualquer imagem aleatória) | `sem_leitura_detectada`, `confianca=0.0` — caso de erro |

## 5. Como o repo é organizado

```
src/leiturista/
├── inference.py      # pipeline det+rec+TrOCR (MeterOCR) — o "modelo"
├── service.py         # ESTE serviço: envelopa MeterOCR num endpoint BentoML
├── cli.py, data.py, train.py, eval.py, artifacts.py  # pipeline de treino/avaliação (fora do escopo desta nota)
├── distribuidora.py   # import do lote real de campo (dados não versionados)
app/app.py              # demo Streamlit (fora do escopo desta nota — front-end)
scripts/gen_sample_images.py  # gera samples/*.png sintéticos
samples/                # imagens/CSV de exemplo, sem dado real
docs/                   # decisões e achados datados
```

## 6. Dados do cliente

Nenhuma foto ou linha de planilha real da distribuidora está neste repositório —
`FotosDistribuidora/`, `data/`, `models/`, `.model_cache/` e `mlflow.db` são
`.gitignore`d, e o histórico do repo foi conferido (`git log --all --diff-filter=A`)
pra confirmar que nunca entraram. As imagens em `samples/` são **sintéticas**,
geradas por `scripts/gen_sample_images.py` (seed fixa, reprodutível); o CSV em
`samples/leituras_exemplo.csv` tem o mesmo schema do CSV real do cliente, com
valores inventados.

## 7. Uso de IA

Este repositório foi construído com apoio do **Claude Code** (Anthropic), em modo
conversacional: pedi para (a) auditar o repo em busca de dado/nome do cliente antes
de qualquer coisa (achou o `docs/CONTEXT_ARCHIVE.txt` e o nome do cliente em 4
arquivos — removido/renomeado), (b) desenhar e escrever o serviço BentoML em cima do
`MeterOCR` já existente, (c) escrever este README e o restante do empacotamento
(`justfile`, `pyproject.toml`, samples sintéticos).

**Avaliação crítica:** o código gerado para `service.py` é deliberadamente curto (um
`predict`, sem abstração extra) e foi lido/testado linha a linha antes de aceitar —
os três testes descritos na seção 4 foram rodados de verdade contra o serviço no ar,
não apenas assumidos. O número mais importante do README (que o TrOCR em produção é
o off-the-shelf, não o fine-tunado) só foi confirmado cruzando 3 documentos
(`inference.py`, `docs/finetune_trocr_ufpramr.md`, `docs/relatorio_benchmark.md`) —
a IA não inventou nem arredondou esse dado.

## Licença

MIT — ver [`LICENSE`](LICENSE).
