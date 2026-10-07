# Roteiro e perguntas prováveis — apresentação SR1 (07/10)

15 min de fala (demo incluída) + 5 min de perguntas. O professor escolhe quem responde,
então todos precisam saber a seção 3. Slides: `leiturista_sr1.pptx`. Plano B: vídeo de 38 s.

## 1. Roteiro por slide

**Slide 1 · Capa (0:15)**
1. Entregamos a camada de serviço: a inferência sai do notebook e vira uma API.
2. Não é produto nem interface. Segue as 4 perguntas do enunciado, nesta ordem.

**Slide 2 · Roteiro (0:45)**
1. Quatro perguntas: modelo, limites, como pôr no ar, demonstração.
2. Escopo: um endpoint `POST /predict`. Sem front-end, sem treino novo, sem nuvem.

**Slide 3 · P1, qual modelo (1:30)**
1. Três modelos públicos, nenhum treinado por nós.
2. PP-OCRv5 mobile detecta as caixas de texto. PP-OCRv6 tiny lê os dígitos (1,1M parâmetros).
   Os dois são da PaddlePaddle, em ONNX, baixados do Hugging Face.
3. TrOCR-small-printed (Microsoft, 62M) é o fallback: lê a imagem inteira quando o detector
   não segmenta o display.
4. Escolha: leves, rodam em CPU, licença aberta. O enunciado avalia o serviço, não o modelo.

**Slide 4 · P1, o caminho de uma foto (1:30)**
1. Foto → detector acha caixas → leitor lê cada caixa → classifica leitura/serial → JSON.
2. Se o display não é segmentado, o TrOCR lê a foto inteira.
3. Tenta também a imagem invertida (display claro em fundo escuro).
4. Flags de coerência: nitidez, várias leituras, sem texto.

**Slide 5 · P2, o gargalo (2:30)**
1. 0,357 de leitura exata e 0,846 por dígito, no UFPR-AMR (300 imagens, display já recortado).
2. ~50% das fotos de campo têm alguma caixa sobre o visor (50 fotos, contagem visual à mão).
3. Consequência: em foto de campo crua o serviço pode ler a placa como se fosse o display.

**Slide 6 · P2, não faz e ajustes (2:00)**
1. Não faz: validar foto contra a ocorrência; `funcao` é heurística; "sem medidor" e
   "ilegível" dão o mesmo valor; confiança é score do OCR, sem calibração.
2. Ajustes: detector de visor treinado com foto de campo; classificador "tem medidor?";
   fine-tune do leitor com recortes de campo; calibrar a confiança.
3. Fora do serviço (10 s, não vale nota): treinamos um CRNN de dígitos (0,873 de leitura
   exata no UFPR-AMR) e o fine-tune do TrOCR. Nenhum dos dois está servido.

**Slide 7 · P3, como pôr no ar (2:00)** (README aberto na tela)
1. `git clone`, `just all`, `just serve`, `just predict`.
2. `just all` faz `uv sync`, baixa os pesos (147 MB, release do GitHub) e gera imagens sintéticas.
3. Python 3.13, dependências fixadas em `uv.lock`. Testado em clone limpo.
4. O serviço leva cerca de 1 min para carregar os modelos: subir ANTES da demo.

**Slide 8 · P4, demonstração (3:30)**
1. Swagger em `localhost:3000` ou `curl -s -F image=@samples/exemplo_01.png localhost:3000/predict`.
2. `exemplo_01`: `leitura_normal`, consumo `017355`, confiança 0,987.
3. `exemplo_03`: leitura + número do medidor (`SN 21868732`).
4. Caso de erro: foto sem medidor → `sem_leitura_detectada`, confiança 0,0, com flags.
5. Se não subir: passar o vídeo, dizer o que quebrou e o que tentou.

**Slide 9 · Fechamento (1:00)**
1. Repositório: README que roda, `uv.lock`, `justfile`, sem segredos, sem dado do cliente.
2. LICENSE (MIT) e seção Uso de IA (Claude Code), declarada no README.

## 2. Divisão de falas sugerida (7 integrantes)

1. Slides 1-2: um integrante. 2. Slides 3-4: dois (modelo, fluxo). 3. Slides 5-6: dois (métricas,
limites). 4. Slide 7: um. 5. Slides 8-9: um faz a demo, outro fecha.
Todos leem a seção 3 antes: qualquer um pode ser chamado.

## 3. Perguntas prováveis e respostas curtas

**Modelo**

1. *Quem treinou o modelo?* Ninguém do grupo. PaddlePaddle (PP-OCR) e Microsoft (TrOCR),
   pré-treinados e públicos. Fizemos um fine-tune do TrOCR no UFPR-AMR, mas o serviço usa o original.
2. *Por que três modelos?* O detector acha onde está o texto, o leitor lê cada caixa, e o TrOCR
   é o fallback quando o detector não segmenta o display.
3. *Por que não usaram o CRNN que treinaram?* Foi treinado só em recortes do UFPR-AMR
   (0,873 no teste) e não foi validado em foto de campo. Não quisemos servir sem essa validação.
4. *O que é o UFPR-AMR?* Base pública de medidores brasileiros (Laroca, IJCNN 2020), 2.000
   imagens já recortadas no display. Usamos para medir o modelo.
5. *Por que PP-OCR e não só TrOCR?* O PP-OCR é bem mais leve (rec com 1,1M parâmetros contra
   62M) e acerta mais no UFPR-AMR (0,357 contra 0,253 de leitura exata).

**Limites**

6. *0,357 é pouco, não?* É, e é exato por foto: tem que acertar todos os dígitos. Por dígito
   é 0,846. Além disso o número é no display já recortado; o problema real é localizar o visor.
7. *O que significa ~50%?* Em 50 fotos inteiras de campo, desenhamos todas as caixas do
   detector e contamos à mão em quantas havia caixa sobre o visor: 25. Amostra pequena, margem
   de ±14 pontos.
8. *Por que o detector falha?* É um detector de texto genérico, não de LCD de segmentos. Falha
   com baixo contraste, névoa e reflexo, e às vezes devolve uma caixa gigante.
9. *O serviço faz todas as funções do projeto?* Não. Lê o consumo e o número do medidor quando
   acha o visor. Não valida se a foto bate com a ocorrência do leiturista (falta rótulo real).
10. *O que melhoraria primeiro?* O detector de visor treinado com foto de campo, porque limita
    tudo que vem depois.

**API e contrato**

11. *O que cada campo da resposta significa?* `numero_medidor` (serial, ou null), `funcao`,
    `consumo` (leitura), `confianca`, `legivel`, `flags` (avisos).
12. *Como `funcao` é decidida?* `sem_leitura_detectada` se nada foi lido; `leitura_normal` se há
    leitura legível com confiança ≥ 0,5; `leitura_incerta` nos demais (vai para conferência manual).
13. *O que é `confianca`?* A maior confiança do OCR entre as caixas de leitura. Vale 0 quando
    não há leitura ou quando a leitura veio do fallback. Não é probabilidade calibrada.
14. *Como é decidido que a foto é legível?* Pela variância do Laplaciano (nitidez).
15. *O que acontece com uma foto sem medidor?* Devolve `sem_leitura_detectada`, confiança 0,0 e
    flags dizendo que não detectou leitura. Não distingue "sem medidor" de "ilegível".

**Pôr no ar e repositório**

16. *Como sobe do zero?* `git clone`, `just all`, `just serve`. Python 3.13 e uv.
17. *Por que BentoML?* É o caminho recomendado no enunciado e já gera o Swagger.
18. *E se não houver internet?* Os pesos ficam em `models/` depois do primeiro `just models`.
19. *Onde estão os dados do cliente?* Não estão no repositório: `.gitignore` cobre os dados, e
    o histórico foi limpo antes da entrega. As imagens de `samples/` são sintéticas.
20. *Como vocês usaram IA?* Claude Code, declarado no README: o que foi pedido e a avaliação
    crítica. Qualquer integrante deve saber explicar qualquer linha.

**Armadilhas (responder com honestidade)**

21. *O serviço está validado em foto real do cliente?* Não. Validamos o contrato e a execução;
    a avaliação em campo foi uma amostra pequena à mão.
22. *Vocês usam o fine-tune do TrOCR?* Não. Existe e está registrado, mas o serviço usa o original.
23. *O classificador "tem medidor?" funciona?* Está preparado, mas não foi treinado. Não
    afirmar resultado.

## 4. Antes de entrar

1. `just serve` rodando e `/readyz` respondendo (cerca de 1 min depois de subir).
2. `samples/` gerado (`just samples`) e README aberto numa aba.
3. Vídeo de plano B aberto em outra janela.
4. Wifi não é necessário: modelos e dependências já baixados.
