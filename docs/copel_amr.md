# Copel-AMR — dataset de leitura de medidores em campo

**Data do registro:** 2026-08-08 | **Status:** NÃO BAIXADO — licença bloqueia
**URL:** https://web.inf.ufpr.br/vri/databases/copel-amr/
**Relevância:** piloto da tarefa de validação de cena (desafio de distribuidora) — ver
`plano_subprojeto_cv.md` §2 e `analise_suficiencia.md`.

---

## O que é

Copel-AMR é um dataset de **12.500 imagens de medidores de energia adquiridas em campo**
pelos leituristas da **Copel** (Companhia Paranaense de Energia, PR) — 395 cidades, 1.113
localidades. Introduzido em:

> R. Laroca, A. B. Araujo, L. A. Zanlorensi, E. C. de Almeida, D. Menotti,
> "Towards Image-based Automatic Meter Reading in Unconstrained Scenarios: A Robust and
> Efficient Approach", **IEEE Access**, vol. 9, pp. 67569-67584, 2021.
> DOI: 10.1109/ACCESS.2021.3077415

### Características

- **Cenário não controlado (unconstrained):** blur (movimento de câmera), sujeira,
  variação de escala, rotação in/out-of-plane, reflexos, sombras, oclusões.
- **~20% (2.500 imgs) sem leitura possível** por oclusão ou medidor defeituoso — útil
  para treinar detecção de "foto inválida/ilegível".
- Resolução: 480×640 ou 640×480 (suficiente para a leitura ser legível quando o medidor
  está íntegro).
- **Split oficial:** 5.000 train / 5.000 test / 2.500 valid (40%/40%/20%).
- **Anotações por imagem:** leitura do medidor, 4 cantos do contador (x,y) (permite
  retificação), bbox (x,y,w,h) por dígito.

### Por que importa para o desafio da distribuidora

É o dataset **mais próximo do cenário real do desafio** entre todos os mapeados:
foto tirada em campo por leiturista (não display isolado em bancada como o UFPR-AMR).
Permite piloto das duas subtarefas de validação de cena:

1. "Esta foto contém um medidor legível?" (segmentação/qualidade — a fração de ilegíveis
   serve como rótulo natural).
2. "A foto corresponde à ocorrência registrada?" — proxy inicial das notas do leiturista.

## Licença (bloqueio atual)

- Dataset é **propriedade da Copel**, liberado **somente** a pesquisadores acadêmicos de
  instituições de ensino/pesquisa, **para fins não-comerciais**.
- Exige preencher o license agreement
  (https://web.inf.ufpr.br/vri/wp-content/uploads/sites/7/2020/07/2020-07-28-Copel-AMR-License-Agreement-1.pdf),
  assinado por **pessoa/autoridade autorizada a comprometer legalmente a instituição**
  (ex.: chefia de departamento, coordenadoria — **aluno ou professor NÃO é aceito**), e
  enviar a **David Menotti** (menotti@inf.ufpr.br).
- **Não há link de download público** enquanto o trâmite não for aceito.
- **Implicação:** para baixar, é preciso acionar a **coordenação da Cesar School** para
  assinar a licença em nome da instituição.

## Próximos passos possíveis

1. Acionar a coordenação/chefia da Cesar School para assinar a licença (necessário para
   o grupo usar o dataset em SR1/SR2).
2. Alternativa imediata sem licença: UFPR-AMR (já baixado) + modelos em `models/`.
3. Alternativas de campo com download aberto: nenhuma confirmada — IEEE DataPort
   (570 imgs, IIT BHU) também exige **login + assinatura** (ver
   `candidatos_nao_baixados.md` §10).

## Referências cruzadas

- Candidatos pendentes: `candidatos_nao_baixados.md` §9 (Copel-AMR) e §2 (UFPR-ADMR-v2,
  também licenciado pela Copel).
- Papel no plano CV: `plano_subprojeto_cv.md` §2 (dados) e §5 (plano de trabalho).
