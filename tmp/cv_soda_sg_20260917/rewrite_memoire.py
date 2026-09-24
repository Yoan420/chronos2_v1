from pathlib import Path
from lxml import etree as E
import zipfile, hashlib

ROOT = Path('C:/Users/BQ6757/chronos2_v1')
SRC = Path('C:/Users/BQ6757/Downloads/Yoan_Kesraoui_CV_These_SODA_NYX.docx')
OUT = ROOT/'output/cv_soda_memoire_sg_20260917/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
source_hash = hashlib.sha256(SRC.read_bytes()).hexdigest()
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{'+NS['w']+'}'
with zipfile.ZipFile(SRC) as z:
    original = {n: z.read(n) for n in z.namelist()}
root = E.fromstring(original['word/document.xml'])
ps = root.findall('.//w:body/w:p', NS)
def text(p): return ''.join(p.xpath('.//w:t/text()', namespaces=NS))
before = [text(p) for p in ps]
title = next(p for p in ps if text(p).startswith('Mémoire | Réseaux feed-forward'))
detail = ps[ps.index(title)+1]
assert 'Applying Deep Learning to Calibrate Stochastic Volatility Models' in text(detail)
new_title = 'Mémoire | Société Générale — Pricing d’options par apprentissage différentiel'
new_detail = '• Étude et implémentation de l’approche de Sridi et Bilokon (2023) : réseaux denses (MLP) entraînés sur les prix de puts européens et leurs sensibilités, puis utilisés pour calibrer le modèle de Heston.'
for paragraph, new, bold in [(title, new_title, True), (detail, new_detail, False)]:
    for child in list(paragraph):
        if child.tag != W+'pPr': paragraph.remove(child)
    run = E.SubElement(paragraph, W+'r')
    props = E.SubElement(run, W+'rPr')
    E.SubElement(props, W+'b').set(W+'val', '1' if bold else '0')
    E.SubElement(props, W+'lang').set(W+'val', 'fr-FR')
    E.SubElement(run, W+'t').text = new
# Correct the duplicate colon in the same SG research mission.
sg = next(p for p in ps if 'Recherche en pricing d’options : : ' in text(p))
extra = next(t for t in sg.findall('.//w:t', NS) if t.text == ': ')
extra.text = ''
after = [text(p) for p in ps]
changed = [i for i, (a,b) in enumerate(zip(before, after)) if a != b]
assert changed == [11, 12, 26], changed
# Retain a single page with the added dissertation entry, preserving font sizes.
for paragraph in ps:
    props = paragraph.find(W+'pPr')
    style = props.find(W+'pStyle') if props is not None else None
    if style is not None and style.get(W+'val') == 'Heading1':
        spacing = props.find(W+'spacing')
        if spacing is None: spacing = E.SubElement(props, W+'spacing')
        spacing.set(W+'before', '100')
        spacing.set(W+'after', '60')
updated = original.copy()
updated['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, value in updated.items(): z.writestr(name, value)
assert hashlib.sha256(SRC.read_bytes()).hexdigest() == source_hash
print(OUT)
print('Updated SG dissertation title and description; fixed duplicate colon. Other content and hyperlinks preserved.')
