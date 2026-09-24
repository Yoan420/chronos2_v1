from pathlib import Path
from copy import deepcopy
from lxml import etree as E
import zipfile, hashlib, json

ROOT=Path('C:/Users/BQ6757/chronos2_v1')
TMP=ROOT/'tmp/cv_soda_sg_20260917'
SRC=ROOT/'output/cv_nyx_engie_20260916/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
OUT=ROOT/'output/cv_soda_sg_20260917/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
EXPECTED='96747f8bc53e21f2f6ba95c8bfb1e00ac9393ad53aef35cb425f49edae167103'
facts=json.loads((TMP/'confirmed_content.json').read_text(encoding='utf-8'))
assert facts['confirmation_received'], 'Wait for the factual research details before finalizing.'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
with zipfile.ZipFile(SRC) as z: original={n:z.read(n) for n in z.namelist()}
root=E.fromstring(original['word/document.xml'])
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W='{'+ns['w']+'}'
body=root.find('w:body',ns); old=list(body)
texts=[''.join(x.xpath('.//w:t/text()',namespaces=ns)) for x in old]
index=next(i for i,t in enumerate(texts) if t.startswith('Société Générale'))
job=old[index]
target=next(x for x in job.findall('.//w:t',ns) if 'Stagiaire en risque de marché quantitatif' in (x.text or ''))
target.text=target.text.replace('Stagiaire en risque de marché quantitatif',facts['role_title'])

template=old[index+2]
bullet=deepcopy(template)
for attr in list(bullet.attrib):
    if attr.endswith('}paraId') or attr.endswith('}textId'):
        del bullet.attrib[attr]
for node in list(bullet):
    if node.tag!=W+'pPr': bullet.remove(node)
for value,bold in [('• ',False),(facts['mission_lead']+' ',True),(facts['mission_detail'],False)]:
    r=E.SubElement(bullet,W+'r'); rp=E.SubElement(r,W+'rPr')
    b=E.SubElement(rp,W+'b'); b.set(W+'val','1' if bold else '0')
    t=E.SubElement(r,W+'t'); t.set('{http://www.w3.org/XML/1998/namespace}space','preserve'); t.text=value
body.insert(index+2,bullet)

updated=original.copy()
updated['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
OUT.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for n,data in updated.items(): z.writestr(n,data)
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
print(OUT)
