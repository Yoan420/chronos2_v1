"""Render the editable Chronos-2 guide as a standalone printable HTML file."""

from __future__ import annotations

from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Guide_Chronos2_de_bout_en_bout.md"
OUTPUT = ROOT / "Guide_Chronos2_de_bout_en_bout.html"


CSS = r"""
:root {
  color-scheme: light;
  --bg: #edf3f8;
  --paper: #ffffff;
  --ink: #172235;
  --muted: #5f6f82;
  --line: #dbe5ee;
  --primary: #0c5b9f;
  --primary-soft: #eaf4fd;
  --accent: #00a78e;
  --accent-soft: #e7f8f4;
  --warning: #9a5c00;
  --warning-soft: #fff5dc;
  --shadow: 0 18px 50px rgba(25, 50, 80, .10);
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0c1524;
  --paper: #111e31;
  --ink: #edf4fb;
  --muted: #aebdd0;
  --line: #2b3a4d;
  --primary: #79bfff;
  --primary-soft: #132f4a;
  --accent: #66d9c6;
  --accent-soft: #123c38;
  --warning: #ffd083;
  --warning-soft: #463514;
  --shadow: 0 20px 60px rgba(0, 0, 0, .28);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.64;
}
.shell {
  display: grid;
  grid-template-columns: minmax(230px, 290px) minmax(0, 980px);
  gap: 34px;
  justify-content: center;
  padding: 30px;
}
.toc {
  position: sticky;
  top: 20px;
  align-self: start;
  max-height: calc(100vh - 40px);
  overflow: auto;
  padding: 22px 20px;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--paper);
  box-shadow: var(--shadow);
  font-size: .86rem;
}
.toc-title {
  margin: 0 0 12px;
  color: var(--primary);
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .09em;
  text-transform: uppercase;
}
.toc ul { margin: 0; padding-left: 17px; }
.toc li { margin: 5px 0; }
.toc ul ul { display: none; }
.toc a { color: var(--muted); text-decoration: none; }
.toc a:hover, .toc a:focus { color: var(--primary); text-decoration: underline; }
article {
  min-width: 0;
  padding: 58px 68px 74px;
  border: 1px solid var(--line);
  border-radius: 22px;
  background: var(--paper);
  box-shadow: var(--shadow);
}
article > h1:first-child {
  margin: -58px -68px 0;
  padding: 68px 68px 8px;
  border-radius: 22px 22px 0 0;
  color: #fff;
  background: linear-gradient(120deg, #073b6f, #087aa3 58%, #00a78e);
  font-size: clamp(2.15rem, 5vw, 4rem);
  line-height: 1.02;
  letter-spacing: -.045em;
}
article > h1:first-child + h2 {
  margin: 0 -68px 30px;
  padding: 12px 68px 48px;
  border-bottom: 0;
  color: rgba(255,255,255,.92);
  background: linear-gradient(120deg, #073b6f, #087aa3 58%, #00a78e);
  font-size: clamp(1.1rem, 2.4vw, 1.65rem);
  font-weight: 500;
}
h2 {
  margin: 56px 0 20px;
  padding-bottom: 10px;
  border-bottom: 2px solid var(--line);
  color: var(--primary);
  font-size: 1.72rem;
  line-height: 1.25;
}
h3 { margin: 34px 0 10px; font-size: 1.23rem; color: var(--ink); }
h4 { margin: 26px 0 8px; }
p, li { max-width: 78ch; }
a { color: var(--primary); }
strong { color: inherit; }
hr { margin: 38px 0; border: 0; border-top: 1px solid var(--line); }
blockquote {
  margin: 24px 0;
  padding: 16px 20px;
  border-left: 5px solid var(--accent);
  border-radius: 0 12px 12px 0;
  background: var(--accent-soft);
}
blockquote p { margin: 0; }
code {
  padding: .12em .36em;
  border-radius: 5px;
  color: var(--primary);
  background: var(--primary-soft);
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: .9em;
}
pre {
  overflow: auto;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #0d1b2a;
  color: #edf6ff;
}
pre code { padding: 0; color: inherit; background: transparent; }
table {
  display: block;
  width: 100%;
  margin: 20px 0 28px;
  overflow-x: auto;
  border-collapse: collapse;
  font-size: .9rem;
}
thead { background: var(--primary-soft); }
th, td { min-width: 92px; padding: 10px 12px; border: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: var(--primary); font-weight: 750; }
tbody tr:nth-child(even) { background: color-mix(in srgb, var(--primary-soft) 32%, transparent); }
.pipeline {
  display: flex;
  align-items: stretch;
  gap: 8px;
  margin: 28px 0 34px;
  overflow-x: auto;
  padding: 4px 2px 14px;
}
.pipeline-step {
  min-width: 150px;
  flex: 1 0 150px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--primary-soft);
  font-size: .84rem;
  line-height: 1.4;
}
.pipeline-step.evaluation { background: var(--warning-soft); }
.pipeline-arrow { align-self: center; color: var(--accent); font-size: 1.5rem; font-weight: 800; }
.actions {
  position: fixed;
  right: 18px;
  bottom: 18px;
  z-index: 10;
  display: flex;
  gap: 8px;
}
.actions button {
  padding: 9px 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--ink);
  background: var(--paper);
  box-shadow: var(--shadow);
  cursor: pointer;
  font: inherit;
  font-size: .82rem;
}
.actions button:hover { border-color: var(--primary); color: var(--primary); }
.meta-note {
  margin-top: 28px;
  padding: 12px 16px;
  border-radius: 10px;
  color: var(--muted);
  background: var(--primary-soft);
  font-size: .82rem;
}
@media (max-width: 1000px) {
  .shell { display: block; padding: 12px; }
  .toc { position: relative; top: 0; max-height: none; margin: 0 auto 14px; max-width: 980px; }
  article { padding: 42px 28px 60px; }
  article > h1:first-child { margin: -42px -28px 0; padding: 50px 28px 8px; }
  article > h1:first-child + h2 { margin: 0 -28px 25px; padding: 10px 28px 36px; }
}
@media print {
  :root { --paper: #fff; --ink: #111; --muted: #444; --line: #bbb; --primary: #174f78; --primary-soft: #eef5f9; --accent-soft: #effaf7; --warning-soft: #fff8e7; }
  @page { size: A4; margin: 16mm 13mm 17mm; }
  body { background: #fff; font-size: 9.6pt; }
  .shell { display: block; padding: 0; }
  .toc, .actions { display: none !important; }
  article { max-width: none; padding: 0; border: 0; box-shadow: none; }
  article > h1:first-child { margin: 0; padding: 34mm 16mm 4mm; border-radius: 0; print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  article > h1:first-child + h2 { margin: 0 0 18mm; padding: 2mm 16mm 24mm; print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  h2 { break-before: auto; break-after: avoid; margin-top: 9mm; }
  h3, h4 { break-after: avoid; }
  p, li { max-width: none; orphans: 3; widows: 3; }
  table { display: table; overflow: visible; break-inside: avoid; font-size: 7.6pt; }
  tr, blockquote { break-inside: avoid; }
  .pipeline { display: grid; grid-template-columns: repeat(2, 1fr); overflow: visible; }
  .pipeline-arrow { display: none; }
  .pipeline-step { min-width: 0; }
  a { color: inherit; text-decoration: none; }
}
"""


SCRIPT = r"""
const root = document.documentElement;
const saved = localStorage.getItem('chronos-guide-theme');
if (saved === 'dark') root.dataset.theme = 'dark';
document.getElementById('theme').addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('chronos-guide-theme', root.dataset.theme);
});
document.getElementById('print').addEventListener('click', () => window.print());
document.getElementById('top').addEventListener('click', () => window.scrollTo({top: 0, behavior: 'smooth'}));
"""


def render() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    converter = markdown.Markdown(
        extensions=("extra", "toc", "sane_lists"),
        extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
        output_format="html5",
    )
    body = converter.convert(source)
    toc = converter.toc
    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'">
<meta name="description" content="Guide non technique de bout en bout du modèle Chronos-2 multi-pays de prévision day-ahead.">
<title>Chronos-2 — Guide de bout en bout</title>
<style>{CSS}</style>
</head>
<body>
<div class="shell">
  <aside class="toc" aria-label="Sommaire"><p class="toc-title">Sommaire</p>{toc}</aside>
  <article>{body}<p class="meta-note">Document généré à partir de <code>{SOURCE.name}</code>. Version 1.0 — 14 août 2026.</p></article>
</div>
<div class="actions" aria-label="Actions du document">
  <button id="theme" type="button" title="Changer de thème">◐ Thème</button>
  <button id="print" type="button" title="Imprimer ou enregistrer en PDF">Imprimer / PDF</button>
  <button id="top" type="button" title="Retour en haut">↑ Haut</button>
</div>
<script>{SCRIPT}</script>
</body>
</html>"""
    OUTPUT.write_text(html, encoding="utf-8", newline="\n")
    print(f"Rendered {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    render()
