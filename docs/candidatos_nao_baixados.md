# Candidatos ainda não baixados — como obter

**Data:** 2026-08-08

Objetivo: imagem de **medidor de distribuição de energia elétrica**. Abaixo os
datasets identificados que ainda não estão na mapen (por exigirem conta/chave/
licença) e o passo-a-passo de obtenção.

## 1. Roboflow — automatic-meter-reading (ENERGIA ELÉTRICA)

- **2.2k a 3.6k imagens** de medidor elétrico, classes `kwh-rating` e `serial no.`, CC BY 4.0
- URL: https://universe.roboflow.com/automatic-meter-reading/automatic-meter-reading
- Precisou: conta Roboflow grátis + API key

```bash
pip install roboflow
python - <<'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="SUA_CHAVE")
p = rf.workspace("automatic-meter-reading").project("automatic-meter-reading")
p.version(6).download("yolov8", location="/Users/joaorietra/Developer/mapen/roboflow_automatic-meter-reading")
EOF
```

## 2. UFPR-ADMR-v2 (ENERGIA ELÉTRICA BR, 5k dials) — licença acadêmica

- Repo: https://github.com/gabriel-salomon/ufpr-admr-v2-dataset
- Preencher `pdfs/license-agreement.pdf`, assinar e enviar a `menotti@inf.ufpr.br`
- Link da licença: https://raw.githubusercontent.com/gabriel-salomon/ufpr-admr-v2-dataset/main/pdfs/license-agreement.pdf

## 3. NRC-GAMMA (gás, 28.8k + 57.7k crops) — download aberto, página SPA

- DOI: https://doi.org/10.4224/3c8s-z290 → objeto no NRC Digital Repository
- URL: https://nrc-digital-repository.canada.ca/eng/view/object/?id=ba1fc493-e65f-4c0a-ab31-ecbcdf00bfa4
- Requer clicar no botão de download na página (repo do Canadá, arquivo zip grande)

## 4. Pointer-10K (10k ponteiros/gauges) — Baidu Pan

- URL: https://pan.baidu.com/s/1R1iZAqKJ2V656EW0RSM3Mg — senha `p10k`
- Requer conta Baidu; licença CC BY-NC-SA 4.0

## 5. SCUT-WMN (água, 5k) — GitHub

- Repo: https://github.com/HCIILAB/Water-Meter-Number-DataSet

## 6. WMeter5K (água, 5k) — GitHub

- Repo: https://github.com/ZZZHANG-jx/WMeter-Reader (branch master)

## 7. Suez MeterReading (água, 1k) — challenge.ens.fr

- URL: https://challengedata.ens.fr/participants/challenges/30/
- Requer conta na plataforma

## 8. Kaggle

- Buscas por "meter reading"/"electricity meter" imagens não retornaram dataset de
  imagem de medidor relevante; só dados tabulares/séries temporais. Imagens exigem
  conta Kaggle de qualquer forma.

## 9. Copel-AMR (ENERGIA ELÉTRICA BR, 12.5k fotos de CAMPO) — alta prioridade

- URL: https://web.inf.ufpr.br/vri/databases/copel-amr/
- **12.500 imagens de campo** (Copel/PR, 395 cidades), cenário não controlado
  (blur, sujeira, rotação, reflexo, sombra, oclusão; ~20% sem leitura legível).
  Dividido 5k train / 5k test / 2,5k valid. Anotações: leitura, 4 cantos do contador,
  bbox por dígito.
- Relevante para o desafio da distribuidora (foto real de campo, não display isolado) —
  usado como piloto da tarefa de validação de cena (ver `plano_subprojeto_cv.md`).
- **LICENÇA CONFIRMADA (2026-08-08):** propriedade da Copel, liberado **somente** a
  pesquisadores acadêmicos p/ uso não-comercial. Exige preencher o license agreement
  (https://web.inf.ufpr.br/vri/wp-content/uploads/sites/7/2020/07/2020-07-28-Copel-AMR-License-Agreement-1.pdf),
  assinado por **autoridade institucional autorizada** (chefia/coordenação — aluno ou
  professor NÃO basta) e enviar a `menotti@inf.ufpr.br`. **Sem esse trâmite não há
  download público.**
- Paper: Laroca et al., "Towards Image-based Automatic Meter Reading in Unconstrained
  Scenarios", IEEE Access 2021, DOI 10.1109/ACCESS.2021.3077415.
- Ação: iniciar o trâmite de licença via Cesar School (coordenação) se o grupo quiser.

## 10. IEEE DataPort — electricity meter readings across diverse conditions

- URL: https://ieee-dataport.org/documents/dataset-electricity-meter-readings-across-diverse-conditions
- 570 JPEGs de medidores em condições diversas (IIT BHU), zip ~91 MB.
- **Acesso (confirmado 2026-08-08): NÃO é download aberto** — página exige
  **login + assinatura IEEE DataPort** ("Subscription Required"). Sem conta/assinatura
  IEEE não há link de download.
- Alternativa: pedir o arquivo aos autores (Hari Prabhat Gupta, Chirag Tank, Mansi
  Dodiya — IIT (BHU) Varanasi) via "Send Author a Private Message" na página.
