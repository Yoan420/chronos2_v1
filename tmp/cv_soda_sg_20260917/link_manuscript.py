from pathlib import Path
from lxml import etree as E
import zipfile

ROOT = Path('C:/Users/BQ6757/chronos2_v1')
SRC = ROOT/'output/cv_soda_sg_20260917/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
OUT = ROOT/'output/cv_soda_nyx_link_20260917/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
TARGET = Path('C:/Users/BQ6757/Downloads/NYX_Research_Manuscript.docx')
assert TARGET.is_file()
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PKG = '{http://schemas.openxmlformats.org/package/2006/relationships}'
ns = {'w': W[1:-1]}
with zipfile.ZipFile(SRC) as z:
    original = {n: z.read(n) for n in z.namelist()}
root = E.fromstring(original['word/document.xml'])
rels = E.fromstring(original['word/_rels/document.xml.rels'])
old_text = root.xpath('//w:t/text()', namespaces=ns)
label = 'NYX for hourly day ahead electricity price forecasting across CWE'
run = next(r for r in root.findall('.//w:r', ns) if ''.join(r.xpath('.//w:t/text()', namespaces=ns)) == label)
ids = {rel.get('Id') for rel in rels}
rid = next('rId' + str(i) for i in range(1, 1000) if 'rId' + str(i) not in ids)
E.SubElement(rels, PKG+'Relationship', Id=rid, Type=R[1:-1]+'/hyperlink', Target=TARGET.as_uri(), TargetMode='External')
link = E.Element(W+'hyperlink')
link.set(R+'id', rid)
link.set(W+'history', '1')
link.set(W+'tooltip', 'Ouvrir le manuscrit NYX (fichier local)')
parent = run.getparent()
index = parent.index(run)
parent.remove(run)
props = run.find(W+'rPr')
if props is None:
    props = E.Element(W+'rPr')
    run.insert(0, props)
for name, value in [('color', '0563C1'), ('u', 'single')]:
    el = props.find(W+name)
    if el is None:
        el = E.SubElement(props, W+name)
    el.set(W+'val', value)
link.append(run)
parent.insert(index, link)
assert old_text == root.xpath('//w:t/text()', namespaces=ns)
updated = original.copy()
for name, node in [('word/document.xml', root), ('word/_rels/document.xml.rels', rels)]:
    updated[name] = E.tostring(node, xml_declaration=True, encoding='UTF-8', standalone=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, value in updated.items():
        z.writestr(name, value)
print(OUT)
print('Hyperlink target:', TARGET.as_uri())
print('All CV text preserved.')
