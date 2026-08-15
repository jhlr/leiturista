# Artigos relacionados — leitura de medidores e mapeamento de redes elétricas

- **Data:** 08/08/2026
- DOIs verificados via Crossref em 08/08/2026.

## 1. Leitura de medidores (AMR — dial/ponteiro)

1. **Laroca, R., et al. (2020).** *Deep Learning for Image-based Automatic Dial
   Meter Reading: Dataset and Baselines.* IJCNN 2020.
   **DOI:** `10.1109/IJCNN48605.2020.9207318` (article IEEE 9207318)
   → Paper **original do dataset UFPR-ADMR** (2k imgs, Copel/PR): dataset público
   real de medidor de ponteiro + baseline de deep learning + análise de erros do
   AMR. **Confirmado como o artigo `9207318` citado na conversa.**

2. **Salomon, G., Laroca, R., Menotti, D. (2022).** *Image-based Automatic Dial
   Meter Reading in Unconstrained Scenarios.* Measurement (Elsevier).
   **DOI:** `10.1016/j.measurement.2022.112025`
   → **UFPR-ADMR-v2**: expandido para 5k imgs (3k/1k/1k), 22.410 anotações;
   YOLOv4 + regressão angular (AngReg); MRR 98.9% (tolerância 1 kWh), MAE 1343→129.
   Contexto: Copel faz >4M leituras/mês, ~850k de medidores de ponteiro.

3. **Zheng, Y., et al. (2021).** *Vector Detection Network: An Application Study on
   Robots Reading Analog Meters in the Wild.* IEEE Transactions on Artificial
   Intelligence. **DOI:** `10.1109/tai.2021.3105936`
   → Paper do dataset **Pointer-10K** (10k ponteiros/gauges, repo
   `DrawZeroPoint/VectorDetectionNetwork`, Baidu `p10k`).

4. **PMIs (citado como "IEEE TIM 2022").** Repo `zzfan3/electric_meter_detect_recognize`
   (1.8k, medidores analógicos de rede/subestação). **Dataset não distribuído
   publicamente** (só código). DOI não confirmado — pendente.

5. **NRC-GAMMA.** NRC Canadá — leitura de medidores analógicos (dataset de **gás**:
   28.8k imgs + 57.7k crops). DOI do dataset: `10.4224/3c8s-z290`. Paper DOI não
   confirmado — pendente.

6. **Smart Grid Stability Prediction** — base do dataset Kaggle *Electrical Grid
   Stability Simulated Data* (tabular; tensão/ângulo/frequência). Referência do
   dataset; não é AMR de imagem.

## 2. Mapeamento geográfico de redes elétricas (power grid maps)

7. **Ahshan, R., Abid, M. S., Al-Abri, M. (2025).** *Geospatial Mapping of
   Large-Scale Electric Power Grids: A Residual Graph Convolutional Network-Based
   Approach with Attention Mechanism.* Energy and AI 20:100486.
   **DOI:** `10.1016/j.egyai.2025.100486`
   → Dataset público em **Zenodo `10.5281/zenodo.14873694`** (CC BY 4.0):
   rede de distribuição de Omã (MZEC: 507k postes, 385k service points, 23.6k
   subestações) + transmissão da Nigéria (~56k componentes). Link-pred 95.9%
   (Omã) / 93.0% (Nigéria). **Baixado** em `data/power_grid_maps/zenodo/`.

8. **Wang, Z., Majumdar, A., Rajagopal, R. (2023).** *Geospatial mapping of
   distribution grid with machine learning and publicly-accessible multi-modal
   data.* Nature Communications 14:5006.
   **DOI:** `10.1038/s41467-023-39647-3`
   → Mapeamento de rede de **distribuição** (EUA) via ML sobre dados multi-modais
   públicos (imagens aéreas). Dataset acompanha o artigo.

## 3. Relacionados (achados na verificação de DOI)

9. **Santos, R. L., et al. (2021).** *Towards Image-Based Automatic Meter Reading in
   Unconstrained Scenarios: A Robust and Efficient Approach.* IEEE Access.
   **DOI:** `10.1109/access.2021.3077415`
10. **Laroca, R., et al. (2019).** *Convolutional neural networks for automatic
    meter reading.* Journal of Electronic Imaging 28(1):013023.
    **DOI:** `10.1117/1.jei.28.1.013023`

## Como verificar/achar DOIs (fluxo)

```bash
# Crossref: procurar por titulo/palavras-chave
curl -s "https://api.crossref.org/works?query.bibliographic=<titulo>&rows=5" | jq -r '.message.items[] | "\(.DOI) | \(.title[0]) | \(.issued["date-parts"][0][0])"'
```

## Relevância pro MAPEN

- **1-2:** base metodológica + dataset BR (UFPR-ADMR v1/v2 via licença acadêmica).
- **3:** abordagem end-to-end com detecção de vetores (pointer) — contraponto ao
  OCR do UFPR.
- **7-8:** referência para a camada de "power grid maps" (contexto geográfico da
  área de estudo) — incluindo como mapear rede a partir de dados públicos.
