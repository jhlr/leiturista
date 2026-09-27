"""Gera a página do Lab 2 a partir dos .md (fonte canônica).

Saídas:
  lab02.html           autossuficiente (doctype + charset), abre por duplo clique
  lab02-artifact.html  derivado do anterior, sem o esqueleto html/head/body, para publicar
"""
import html
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = Path(__file__).resolve().parent

LINKS = {
    "guia-arquitetura-rede-profunda.md": "#guia",
    "lab02-enunciado.md": "#enunciado",
    "codigo/esqueleto_arquitetura.py": "#codigo",
}


def converter(arquivo: str, prefixo: str):
    texto = (RAIZ / arquivo).read_text(encoding="utf-8")
    for alvo, ancora in LINKS.items():
        texto = texto.replace(f"]({alvo})", f"]({ancora})")
    texto = re.sub(r"^# .*\n", "", texto, count=1)
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"slugify": lambda v, s: f"{prefixo}-{slugify_unicode(v, s)}",
                                   "toc_depth": "2"}},
    )
    corpo = md.convert(texto)
    corpo = corpo.replace("<table>", '<div class="tabela"><table>').replace("</table>", "</table></div>")
    return corpo, md.toc_tokens


def sumario(tokens):
    itens = "".join(f'<li><a href="#{t["id"]}">{t["name"]}</a></li>' for t in tokens if t["level"] == 2)
    return f'<ol class="sumario">{itens}</ol>'


CSS = (SAIDA / "estilo.css").read_text(encoding="utf-8")

enunciado, toc_enun = converter("lab02-enunciado.md", "e")
guia, toc_guia = converter("guia-arquitetura-rede-profunda.md", "g")
codigo = html.escape((RAIZ / "codigo/esqueleto_arquitetura.py").read_text(encoding="utf-8"))

conteudo = f"""<title>Arquitetura Profunda Lab 2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
{CSS}
</style>
<header class="topo">
  <div class="topo-inner">
    <p class="rotulo">Deep Learning · BD e IA · CESAR School · Projeto 4 distribuidora</p>
    <h1>Laboratório 2<br><span>A arquitetura profunda do seu projeto</span></h1>
    <div class="registro" aria-label="Leitura de exemplo: 04821 quilowatt-hora">
      <span>0</span><span>4</span><span>8</span><span>2</span><span>1</span><em>kWh</em>
    </div>
    <dl class="ficha">
      <div><dt>Valor</dt><dd>1,0 ponto</dd></div>
      <div><dt>Entrega</dt><dd>por grupo</dd></div>
      <div><dt>Prazo</dt><dd>08/10/2026, 23h59</dd></div>
      <div><dt>Pasta</dt><dd><code>lab-02/</code></dd></div>
    </dl>
  </div>
</header>
<nav class="abas" aria-label="Seções">
  <a href="#enunciado">Enunciado</a>
  <a href="#guia">Guia de arquitetura</a>
  <a href="#codigo">Código de referência</a>
</nav>
<main>
  <section id="enunciado" class="doc">
    <p class="rotulo">Parte 1</p>
    <h1 class="titulo-doc">Enunciado</h1>
    <details class="indice" open><summary>Nesta parte</summary>{sumario(toc_enun)}</details>
    {enunciado}
  </section>
  <section id="guia" class="doc">
    <p class="rotulo">Parte 2 · material de apoio</p>
    <h1 class="titulo-doc">Como montar a arquitetura de uma rede profunda</h1>
    <details class="indice" open><summary>Nesta parte</summary>{sumario(toc_guia)}</details>
    {guia}
  </section>
  <section id="codigo" class="doc">
    <p class="rotulo">Parte 3</p>
    <h1 class="titulo-doc">Código de referência</h1>
    <p><code>esqueleto_arquitetura.py</code> roda em CPU, sem baixar nada:
    <code>python esqueleto_arquitetura.py</code>. Com <code>--pretreinado</code>, baixa os pesos
    ImageNet da MobileNetV3. Copie o arquivo abaixo para o repositório do grupo.</p>
    <pre class="fonte"><code>{codigo}</code></pre>
  </section>
</main>
<footer class="rodape">Deep Learning · Tecnológico em Banco de Dados e IA · 4º período · 2026.2</footer>
"""

autossuficiente = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{conteudo.split('<header', 1)[0]}</head>
<body>
<header{conteudo.split('<header', 1)[1]}</body>
</html>
"""
(SAIDA / "lab02.html").write_text(autossuficiente, encoding="utf-8")

cabeca = re.search(r"<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\"[^>]*>\n(.*)</head>", autossuficiente, re.S).group(1)
corpo = re.search(r"<body>\n(.*)</body>", autossuficiente, re.S).group(1)
(SAIDA / "lab02-artifact.html").write_text(cabeca + corpo, encoding="utf-8")
print("ok:", len(autossuficiente), "bytes")
