# Esquema de rótulos — Luminus (Lab 2, item C1)

Unidade: **uma foto inteira**. Cada foto recebe uma `classe` e, quando legível, a `leitura`.
Planilha por rotulador: `nome_arquivo, lote, rotulador, classe, leitura, observacao`.
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
