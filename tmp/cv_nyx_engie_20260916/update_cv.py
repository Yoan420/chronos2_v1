from pathlib import Path
from lxml import etree as E
import zipfile, hashlib, json

ROOT=Path('C:/Users/BQ6757/chronos2_v1')
SRC=ROOT/'output/cv_nyx_20260916/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
OUT=ROOT/'output/cv_nyx_engie_20260916/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
EXPECTED='4763d10026b0566282007a5b5705886022db45e0d62a69aa65441b70a8b7fd60'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED,'Source changed, rebase first.'
with zipfile.ZipFile(SRC) as z: original={n:z.read(n) for n in z.namelist()}
root=E.fromstring(original['word/document.xml'])
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
body=root.find('w:body',ns); p=list(body)
assert len(p)==39
before={i:E.tostring(n) for i,n in enumerate(p)}
def text(n): return ''.join(n.xpath('.//w:t/text()',namespaces=ns))
def change_text(n,old,new):
    nodes=n.findall('.//w:t',ns)
    assert ''.join(x.text or '' for x in nodes).count(old)==1
    # Existing titles occupy a single text run; preserve formatting and date run.
    target=next(x for x in nodes if old in (x.text or ''))
    target.text=target.text.replace(old,new)

change_text(p[7],'NYX  |  Prévision des prix de l’électricité à J+1',
    'NYX  |  Travaux réalisés en stage chez ENGIE Global Markets')
change_text(p[20],'ENGIE Global Markets  |  Stagiaire en recherche quantitative',
    'ENGIE Global Markets  |  ML/DL Research - Power Market')
change_text(p[21],'Global Market Analysis, Paris','Stage · Global Market Analysis, Paris')

# Attribute design and experimental results to the ENGIE internship. Keep the
# manuscript citation in Research and remove the duplicate generic ENGIE bullet.
new=[]
for i,n in enumerate(p):
    if i in (9,10,22): continue
    new.append(n)
    if i==21: new.extend([p[9],p[10]])
for n in list(body): body.remove(n)
for n in new: body.append(n)

for i,n in enumerate(p):
    if i not in (7,20,21,22):
        assert E.tostring(n)==before[i],f'Unintended change in paragraph {i}'
assert text(p[1])=='Candidature '
assert 'KNN' in text(p[25])
assert 'Conception de NYX, modèle hybride de prévision' not in text(body)
assert 'Stagiaire en recherche quantitative' not in text(body)
assert text(body).count('Conception et formalisation de NYX')==1

updated=original.copy()
updated['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
OUT.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in updated.items(): z.writestr(name,data)
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
audit={'source_sha256':EXPECTED,'source_preserved':True,
       'changed_part':'word/document.xml','changed_original_paragraphs':[7,20,21],
       'moved_paragraphs_under_engie':[9,10],'removed_duplicate_generic_bullet':22,
       'manual_user_edits_preserved':True,'output':str(OUT)}
(ROOT/'tmp/cv_nyx_engie_20260916/edit_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
