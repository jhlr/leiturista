# Projetos e repositórios notáveis — leitura de medidores (levantamento inicial)

- **Data:** 08/08/2026
- **Origem:** levantamento do início da sessão (HF, Kaggle, GitHub, acadêmico) — os
  "github e projetos notáveis" citados na conversa.
- **Contexto:** imagem de medidor de energia elétrica **não tem** dataset grande e
  curado no Kaggle nem no HF. O que existe de imagem real está em repositórios
  acadêmicos/GitHub — ver lista abaixo.

## 1. GitHub / acadêmico (os principais)

| Projeto | Repo / URL | Conteúdo | Licença / acesso | Relevância p/ MAPEN |
|---|---|---|---|---|
| **UFPR-ADMR-v2** | `github.com/gabriel-salomon/ufpr-admr-v2-dataset` | 5.000 imgs de **medidor de ponteiro (dial)** de energia elétrica, **Copel/Paraná**; 3k train / 1k val / 1k test; 22.410 anotações (cantos do display + dials + leitura) | Licença acadêmica (não-comercial): assinar `pdfs/license-agreement.pdf` e enviar a `menotti@inf.ufpr.br` (~1-5 dias úteis) | **★ Melhor p/ o caso BR** (distribuição, mesmo padrão Copel) |
| **UFPR-ADMR-v1** | `web.inf.ufpr.br/vri/databases/ufpr-admr/` | 2k dials, mesma família Copel | Termo: `web.inf.ufpr.br/vri/wp-content/uploads/sites/7/2020/03/UFPR-ADMR-License-Agreement.pdf` (mesmo fluxo Prof. Menotti) | ★ família BR |
| **PMIs** | `github.com/zzfan3/electric_meter_detect_recognize` | 1.8k, medidores analógicos de rede/subestação (paper IEEE TIM 2022) | Repo só tem **código — dataset não distribuído publicamente** | referência de método |
| **Pointer-10K** | `github.com/DrawZeroPoint/VectorDetectionNetwork` | 10k ponteiros/gauge | Baidu Pan `pan.baidu.com/s/1R1iZAqKJ2V656EW0RSM3Mg` (senha `p10k`); CC BY-NC-SA 4.0 | ponteiros (transferível p/ dial) |
| **SyntheticGauges + RealGauges** | `jjcvision.com/projects/gauge_reading.html` | 11k, inclui sintético | Google Drive `drive.google.com/drive/folders/1j2sKwcqHCfVJ1BK6Rj6DT41rSCmXdPxT` (CC BY-NC 4.0) — **link quebrado/404 na verificação** | gauge (transferível) |
| **NRC-GAMMA** | `github.com/nrc-cnrc/NRC-GAMMA` | **gás** analógico: 28.8k imgs + 57.7k crops | DOI `doi.org/10.4224/3c8s-z290` → `nrc-digital-repository.canada.ca/eng/view/object/?id=ba1fc493-e65f-4c0a-ab31-ecbcdf00bfa4` | transferível p/ elétrico |
| **Awesome list (curadoria)** | `github.com/ZZZHANG-jx/Awesome-Image-based-Meter-Recognition-Reading` | **Lista completa** de datasets públicos de reconhecimento de medidor/imagem | — | fonte de caça futura |

## 2. Hugging Face (datasets com imagem)

| Dataset | Conteúdo | Licença | Status |
|---|---|---|---|
| `Ishtiak113/Electricmeter` (+ cópia `Taj207/Electricmeter`) | imagefolder <1k, medidor elétrico (zip 23.6 MB) | própria ("rafi"), sem curadoria | fotos do usuário (`~/Downloads/Electric Meter Photos`, 119 jpg) vieram deste zip |
| `Chaymaa/meter_reading` | amostra 6 imgs (imagem+texto) | — | **baixado** (raiz `mapen/`, 3 parquets) |
| `Chaymaa/UFPR-AMR` | **2.000 imgs** de medidor elétrico BR (Elster, placa INMETRO/DIMEL) — o real | — | **baixando** (train/valid pendentes) |
| `goodcoffee/Meter_Reading` | 498 png + JSON de VQA | Apache-2.0 | baixado |
| `ud-smart-city/water-meter-image` | **água**: 20 fotos + bbox + máscaras | CC BY-NC-ND 4.0 | baixado |
| Modelo `Word2Li/Electricity-Meter-OCR-7B` | Qwen2.5-VL-7B finetunado p/ leitura end-to-end (841 imgs privadas de curso) | MIT | documentado em `word2li_electricity-meter-ocr-7b.md`; download cancelado |

## 3. Kaggle

- **Nada de imagem de medidor.** O que aparece com "meter reading" é tabular
  (`coldblot/meter-reading-dataset`, `thomaszengerle/mains-voltage-readings-smart-meter`).

## 4. Outros candidatos (exigem conta/chave ou pedem acesso)

Roboflow `automatic-meter-reading` (2.2k–3.6k imgs, classes `kwh-rating`/`serial no.`,
CC BY 4.0, precisa API key), **SCUT-WMN** e **WMeter5K** (medidores de água chineses),
**Suez meter reading challenge** (`challenge.ens.fr`, França).
→ passo-a-passo em `candidatos_nao_baixados.md`.

## 5. Resumo da situação

- **Energia elétrica** com download imediato: só `UFPR-AMR` (2k, em andamento) + Praekelt (165).
- **Energia elétrica BR de verdade (dial/Copel):** `UFPR-ADMR-v2` (5k) — vale pedir a
  licença acadêmica (ação de ~5 min + e-mail; dá o maior ganho de volume/qualidade BR).
- Ponteiros/gauges/gás (NRC-GAMMA, Pointer-10K, Synthetic/RealGauges) são transferíveis
  mas não são medidor de distribuição BR.
