# Esquema de rótulos — Luminus (Lab 2, item C1)

Unidade: **uma foto inteira**. Cada foto recebe uma `classe` e, quando legível, a `leitura`.
Planilha por rotulador: `nome_arquivo, lote, rotulador, classe, leitura, observacao`. Arquivo final da entrega:
[`rotulos.csv`](rotulos.csv) (colunas na seção "Fontes e arquivo final").
As fotos não ficam no repositório (dado do cliente); os exemplos abaixo são só nomes de arquivo.

## Classes

| Classe | Definição operacional | `leitura` |
|---|---|---|
| `legivel` | Existe um display (LCD, ou tambor/contador mecânico) em que **é possível transcrever a sequência de dígitos** sem adivinhar nenhum dígito. | os dígitos lidos, sem separador, sem o prefixo de item do display (ex.: o "03" à esquerda no Cronos), sem zeros inventados |
| `ilegivel` | Existe medidor na foto, mas **não se transcreve a leitura**: display borrado, apagado, com reflexo ou brilho cobrindo os dígitos, escuro demais, distante demais, coberto por grade/sujeira, ou mostrador analógico de ponteiros. | vazio |
| `sem_medidor` | **Nenhum medidor** identificável na foto (fachada, muro, portão, caixa vazia, objeto qualquer). | vazio |

Regras de desempate (valem para os dois rotuladores):

1. Um dígito duvidoso torna a foto `ilegivel`; não se completa dígito "pelo contexto".
2. Se dá para ver que existe uma caixa/medidor mas o display não aparece (reflexo, sombra, ângulo), é `ilegivel`, não `sem_medidor`.
3. Contador mecânico com todos os dígitos nítidos é `legivel`; dígito intermediário entre dois números (tambor em transição) é `ilegivel`.
4. Foto com vários medidores: rotule o que ocupa mais área; se empatar, o mais nítido.

## Exemplos-limite (2 por classe)

| Classe | Quase é | Quase não é |
|---|---|---|
| `legivel` | `PSP_EXTRATLEITIMPL_220526_0408_20000000017302980975_000.jpg`: medidor atrás de vidro com reflexo, contador "ELO 2106 D" ainda transcrevível | `PSP_EXTRATLEITIMPL_210526_0335_20000000008052964749_000.jpg`: LCD visível mas o reflexo cobre parte do display; só vale `legivel` se todos os dígitos aparecerem |
| `ilegivel` | `PSP_EXTRATLEITIMPL_030726_0121_20000000016002990371_000.jpg`: display visível, mas pequeno e sem contraste; dígitos não se transcrevem sem chute | `PSP_EXTRATLEITIMPL_030726_0121_20000000004903012106_000.jpg`: barra e reflexo cobrem o display; há medidor, então não é `sem_medidor` |
| `sem_medidor` | `PSP_EXTRATLEITIMPL_220526_0408_20000000018602990282_000.jpg`: caixa de medição com o reflexo do fotógrafo; se o medidor não aparece sob o reflexo, `sem_medidor`, se aparece o contorno, `ilegivel` | `PSP_EXTRATLEITIMPL_030726_0121_20000000000502987344_000.jpg`: grades e medidores muito ao fundo, sem display visível; decidir pela regra 2 |

## Como rotular

`LEITURISTA_FOTOS_DIR=<pasta das fotos> streamlit run app/rotular.py` (cada integrante vê só o seu bloco;
os 50 pares em dupla são sorteados em `lab-02/rotulos/plano.csv`, com rotuladores diferentes e sem ver o rótulo do outro).
Depois: `leiturista rotulos-kappa` une as planilhas e calcula o kappa de Cohen por classe. Kappa < 0,6 em alguma
classe obriga a reescrever a definição e registrar a mudança no `README.md`.


## Fontes e arquivo final (`rotulos.csv`)

Duas fontes rotularam a **classe**; a **leitura** vem de uma terceira, o campo.

| Fonte | Quem | Fotos | Observação |
|---|---|---|---|
| `mikael` | humano, triagem em 3 pastas (`medidor`, `medidoruim`, `ntem`) mapeadas para `legivel`, `ilegivel`, `sem_medidor` | 3.063 (só o lote `220526_0408`) | sem leitura transcrita |
| `llama32v` | Llama-3.2-Vision 11B local, convertido: medidor não visível = `sem_medidor`; visível e display legível = `legivel`; visível e não legível = `ilegivel` | 3.247 (4 lotes) | rótulo fraco, leniente |
| leitura | o leiturista digitou em campo (`Posicao do medidor lida` nos `BaseExtracao_*.csv` de cada lote) | 11.475 | **não foi conferida na foto** |

Colunas de `rotulos.csv`: `nome_arquivo, lote, split, rotulador, classe, classe_humano, classe_llm, leitura, leitura_origem`.

1. `classe` = **união**: `legivel` se pelo menos uma fonte disser `legivel`; senão `ilegivel` se alguma disser `ilegivel`; senão
   `sem_medidor`. É uma escolha de recall ("vale tentar ler"), usada para treino e triagem.
2. `classe_humano` e `classe_llm` ficam separadas para trocar a regra sem refazer o arquivo. **Para avaliar um modelo, o rótulo de
   referência é `classe_humano`** quando existir; a união não é verdade-terreno.
3. `leitura` só é preenchida quando `classe = legivel`, copiada da digitação do leiturista (`leitura_origem =
   digitada_leiturista`). Pode estar errada mesmo com a foto legível; ninguém a transcreveu olhando a foto.
4. `split` por lote: treino `200526_0352` e `210526_0335`, valid `220526_0408`, teste `030726_0121` (ver C4 no `README.md`).
   O rótulo humano está todo no lote de valid; treino e teste têm só o rótulo do LLM.

## O que mudou depois da medição de concordância (kappa)

Segunda rotuladora: o Llama-3.2-Vision, em 688 fotos que os dois rotularam (todas do lote `220526_0408`). Concordância 0,562.

| Classe (um contra todos) | Kappa |
|---|---|
| `legivel` | 0,285 |
| `ilegivel` | 0,239 |
| `sem_medidor` | 0,778 |

`legivel` e `ilegivel` ficam **abaixo de 0,6**: a definição estava ambígua. O desacordo é de um tipo: em 276 fotos o modelo
disse legível e o humano ilegível (o oposto, 1). Nessas 276, o modelo classificou a qualidade como `good` em 207 (75%), só 22
`occluded`, 19 `blurry`, 18 `dark`.

**Auditoria visual (12 fotos sorteadas das 276, semente 1, miniaturas de ~270 px; amostra pequena, leitura a olho):**

| Veredito olhando a foto | Fotos | Quem errou |
|---|---|---|
| Ilegível de fato (grade na frente, vidro quebrado, display minúsculo, reflexo, mecânico sujo) | 5 | o modelo foi leniente |
| Borderline (display pequeno ou reflexo, sequência parcial) | 2 | indefinido |
| Parecem legíveis (LCD nítido com a sequência inteira, ou tambor ELO 2106 D com os dois registros) | 5 | o humano foi mais rígido |

A hipótese anterior ("display nítido, mas sequência incompleta") **não se sustentou**: nas 5 fotos que parecem legíveis, a sequência
aparece inteira nas miniaturas. O modelo aceita foto com **grade, vidro trincado, display pequeno ou reflexo cobrindo**. Já as
5 fotos que pareciam legíveis foram rotuladas ilegíveis pelo Mikael; o autor confirmou que a pasta `medidoruim` significa
**ilegível** (não "medidor em mau estado"), então não é erro de mapeamento das pastas. Limite da auditoria: foi feita em
miniaturas de ~270 px e a olho; em resolução cheia, pode ser que esses dígitos não se transcrevam sem chute, que é o critério
da regra 1. Esse lado só se decide com um segundo humano na mesma foto (C3 humano × humano).

Mudanças no esquema:
1. **Nova regra 5:** obstáculo físico sobre o display (grade, vidro trincado, reflexo cobrindo parte dos dígitos) ou display
   ocupando pouco da foto, de modo que nenhum dígito se separa do vizinho, é `ilegivel`, ainda que o resto da foto seja nítido.
2. **Nova regra 6:** em conflito entre fontes, vale o humano para avaliação; a união só serve a treino e triagem (ver acima).
3. A regra 1 continua; `sem_medidor` não mudou (kappa 0,78).

Regras de desempate acrescentadas:

5. Obstáculo sobre o display, ou display minúsculo na foto: `ilegivel`.
6. Divergência entre fontes: referência de avaliação = humano; treino/triagem = união.
