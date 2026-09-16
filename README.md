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

Testado com `uv` (resolve o Python sozinho, 3.13). Tempo aproximado: ~2 min (`uv sync`
baixa/instala ~160 pacotes) + ~2 min (`just models`, 147 MB) + a primeira chamada de
`bentoml serve` demora ~15-30s pra abrir a porta (import "frio" de torch/transformers)
numa rede razoável.

```bash
git clone https://github.com/jhlr/leiturista.git && cd leiturista
just setup      # uv sync — instala as dependências (uv.lock)
just models     # baixa + extrai os pesos (release público modelos-1.0, 147 MB)
just samples    # gera samples/*.png (sintéticas, seed fixa — não versionadas)
just serve      # sobe o serviço em http://localhost:3000
```

Sem `just` instalado: `brew install just` (macOS) ou `cargo install just`. Os
comandos equivalentes sem `just`:

```bash
uv sync
curl -L https://github.com/jhlr/leiturista/releases/download/modelos-1.0/leiturista-models.tar.gz -o leiturista-models.tar.gz
tar -xzf leiturista-models.tar.gz
uv run python scripts/gen_sample_images.py
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
geradas por `scripts/gen_sample_images.py` (seed fixa, reprodutível) — não são
versionadas (`samples/*.png` no `.gitignore`), justamente por serem regeráveis em
segundos com `just samples`; o `samples/leituras_exemplo.csv` (autoral, versionado)
tem o mesmo schema do CSV real do cliente, com valores inventados.

## 7. Uso de IA

Ferramenta: **Claude Code** (Anthropic, Sonnet 5), em modo conversacional, com
acesso a shell, edição de arquivos e GitHub CLI. Detalhado abaixo porque "código
gerado que ninguém da equipe sabe explicar conta como não entregue" — este é o
registro do que foi pedido e a avaliação crítica de cada resposta.

### O que foi pedido, em ordem

1. Revisão de um PR anterior (import do lote real de fotos de campo) — aceitar tudo
   exceto um arquivo de snapshot de sessão que não deveria ir pro repo público.
2. Implementação do SR1 a partir do enunciado do professor, colado inteiro no chat.
3. Três ajustes pontuais durante a implementação: trocar o import do PIL
   (`from PIL import Image as pil`); autorização explícita para testar contra fotos
   reais de campo localmente (sem commitar), em vez de só sintéticas; manter
   `service.py` legível, sem abstração extra em cima do `MeterOCR`.
4. Este relatório de uso de IA, dentro do próprio README.

### O que a IA fez por iniciativa própria (não pedido explicitamente)

- Antes de tocar em qualquer arquivo, auditou se o repositório já era público e se
  continha nome do cliente ou dado real commitado — achou o nome do cliente em 4
  arquivos rastreados (removidos/renomeados) e confirmou via
  `git log --all --diff-filter=A` que nenhuma foto/CSV real tinha sido commitada em
  nenhum momento do histórico.
- Apurou de qual MLflow run vem o TrOCR carregado em produção hoje, cruzando
  `inference.py`, `docs/finetune_trocr_ufpramr.md` e `docs/relatorio_benchmark.md` —
  descobriu que é o checkpoint **off-the-shelf** (`9c14db62`), não o fine-tunado
  como um doc mais antigo sugeria, e reportou isso sem arredondar pra cima.

### Avaliação crítica

**Onde funcionou bem:** o `service.py` gerado é curto (uma classe, um método
`predict`) e foi lido linha a linha antes de aceitar. Todo `curl` deste README foi
de fato executado contra o serviço no ar (imagem pública UFPR-AMR, as 4 sintéticas
de `samples/`, e uma foto real de campo só pra validar a seção de limites, sem
commitar o resultado) — nenhum exemplo é hipotético. Os números de exact-match/
digit-acc citados vieram de doc já existente no repo, não foram inventados.

**Onde precisou de correção humana:** no merge do PR anterior, a IA empurrou a
remoção de um arquivo pra uma branch nova em `origin` em vez de atualizar a branch
do fork de origem do PR — o squash merge trouxe o arquivo de volta pro `main`,
precisando de um segundo commit corretivo depois que o erro foi percebido (registrado
aqui porque é o tipo de erro que passa batido se ninguém conferir o resultado). O
plano inicial também só previa imagens sintéticas pros testes de limite — foi pedido
explicitamente rodar também contra fotos reais de campo antes de escrever a seção 2.
Estilo de import do PIL foi corrigido por pedido direto, não por iniciativa da IA.

## Licença

MIT — ver [`LICENSE`](LICENSE).
