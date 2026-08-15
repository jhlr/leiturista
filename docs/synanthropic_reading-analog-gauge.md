# Synanthropic/reading-analog-gauge — como baixar

**Data:** 2026-08-08

## O que é

Dataset de **dial/gauges analógicos genéricos** (leitura de ponteiro em mostradores
industriais). Contém dois sub-datasets de anotação — `corner` e `keypoint` — usados
para treinar detecção de cantos/pontos-chave do ponteiro. Não é específico de
medidor de **energia elétrica**: é gauge genérico (pressão, temperatura etc.).

- HF: https://huggingface.co/datasets/Synanthropic/reading-analog-gauge
- Origem/versão canônica: https://huggingface.co/datasets/Synanthropic/reading-analog-dial
- Demo do modelo: https://huggingface.co/spaces/Synanthropic/reading-analog-dial
- Tamanho: ~2.6 GB (6 arquivos, ~10K-100K imagens)

## Por que não foi baixado

Foco atual do projeto = imagem de **medidor de distribuição de energia elétrica**.
Este dataset é de gauges analógicos genéricos, então ficou fora do download automático.

## Como baixar (quando quiser)

Com `huggingface_hub` (CLI):

```bash
HF=/var/folders/q7/bkpgntz547n9gc64szcfxpm40000gn/T/opencode/pqenv/bin/hf
$HF download Synanthropic/reading-analog-gauge \
  --repo-type dataset \
  --local-dir /Users/joaorietra/Developer/mapen/Synanthropic_reading-analog-gauge
```

Ou via Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Synanthropic/reading-analog-gauge",
    repo_type="dataset",
    local_dir="/Users/joaorietra/Developer/mapen/Synanthropic_reading-analog-gauge",
)
```

## Pré-processamento (README original)

1. Baixar e descompactar um dos sub-datasets (`corner` ou `keypoint`)
2. Editar `make-dataset.py` → `srcdir` apontando para `corner` ou `keypoint`
3. Rodar o script e treinar o modelo de leitura de ponteiro
