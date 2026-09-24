from pathlib import Path
from copy import deepcopy
from lxml import etree as E
import hashlib, zipfile, json

ROOT=Path('C:/Users/BQ6757/chronos2_v1')
SRC=ROOT/'output/cv_these_75968/Yoan_Kesraoui_CV_These_SODA.docx'
OUT=ROOT/'output/cv_nyx_20260916/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
EXPECTED='1a178e102925a73d649c025b37ebdbfc629542f1535461dba3c04e936dd16b9c'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W='{'+NS['w']+'}'
with zipfile.ZipFile(SRC) as z: parts={n:z.read(n) for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']); body=root.find('w:body',NS)
old=list(body); p=old[:-1]
assert len(p)==37 and not root.findall('.//w:tbl',NS)

def text(n): return ''.join(n.xpath('.//w:t/text()',namespaces=NS))

def replace_para(template, chunks):
    new=deepcopy(template)
    run=template.find('w:r',NS)
    for c in list(new):
        if c.tag!=W+'pPr': new.remove(c)
    for value,bold,italic,size in chunks:
        r=deepcopy(run) if run is not None else E.Element(W+'r')
        for c in list(r):
            if c.tag!=W+'rPr': r.remove(c)
        rp=r.find('w:rPr',NS)
        if rp is None: rp=E.Element(W+'rPr'); r.insert(0,rp)
        for name,v in [('b',bold),('i',italic),('sz',size)]:
            if v is not None:
                el=rp.find('w:'+name,NS)
                if el is None: el=E.SubElement(rp,W+name)
                el.set(W+'val',str(v if name=='sz' else int(v)))
        values=value.split('\t')
        for i,seg in enumerate(values):
            if i: E.SubElement(r,W+'tab')
            t=E.SubElement(r,W+'t'); t.set('{http://www.w3.org/XML/1998/namespace}space','preserve'); t.text=seg
        new.append(r)
    return new

def plain(template,value,bold=None,italic=None,size=None):
    return replace_para(template,[(value,bold,italic,size)])

def property(p,name,**attrs):
    pp=p.find('w:pPr',NS)
    if pp is None: pp=E.Element(W+'pPr'); p.insert(0,pp)
    el=pp.find('w:'+name,NS)
    if el is None: el=E.SubElement(pp,W+name)
    for k,v in attrs.items(): el.set(W+k,str(v))

profile=plain(p[6],
    'Formation en data science, économétrie et mathématiques. Conception de NYX, modèle hybride de prévision fondé sur Chronos-2 ; évaluation multi-marchés et rédaction scientifique en anglais.')
heading=plain(p[13],'TRAVAUX DE RECHERCHE')
nyx=replace_para(p[18],[
    ('NYX  |  Prévision des prix de l’électricité à J+1',True,False,None),
    ('\t2026',False,False,19)])
property(nyx,'spacing',before=0,after=20)
citation=replace_para(p[19],[
    ('Kesraoui, Y. (2026). ',False,False,19),
    ('NYX for hourly day ahead electricity price forecasting across CWE. ',False,True,19),
    ('Manuscrit non publié.',False,False,19)])
property(citation,'keepNext',val=1)
methods=replace_para(p[20],[
    ('• ',False,False,None),
    ('Conception et formalisation de NYX',True,False,None),
    (' : modèle de fondation Chronos-2 gelé, correction des résidus par CatBoost et filtrage adaptatif ; étude des interactions entre les composantes.',False,False,None)])
results=replace_para(p[20],[
    ('• Évaluation rétrospective sur ',False,False,None),
    ('4 marchés et 350 jours',True,False,None),
    (' : ',False,False,None),
    ('MAE réduite de 8,27 % face à Chronos-2 seul',True,False,None),
    (' ; validation chronologique et analyse de la calibration et des prix extrêmes.',False,False,None)])
memo_title=plain(p[14],'Mémoire  |  Transformers et modèles de fondation pour les séries temporelles')
property(memo_title,'spacing',before=60,after=40)
property(memo_title,'keepNext',val=1)
memo=plain(p[15],
    '• Revue des Transformers temporels (Chronos-2, TimesFM, Moirai) : attention, préentraînement, transfert, fuites de données et limites des prévisions probabilistes.')
engie=plain(p[20],
    '• Développement de pipelines Python de prévision à J+1 ; validation glissante, contrôle des fuites de données et analyse des biais, des événements extrêmes et des régimes de marché.')

research=[heading,nyx,citation,methods,results,memo_title,memo]
experience=[engie if i==20 else p[i] for i in range(17,37) if i!=21]
new=p[:6]+[profile]+research+p[7:13]+experience+[old[-1]]
for c in list(body): body.remove(c)
for c in new: body.append(c)
out=parts.copy()
out['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
OUT.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in out.items(): z.writestr(name,data)

unchanged=[i for i in range(37) if i not in (6,13,14,15,16,20,21)]
assert all(text(p[i]) in [text(n) for n in new] for i in unchanged)
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
audit={'source_sha256':EXPECTED,'source_unchanged':True,
       'only_modified_package_part':'word/document.xml',
       'new_paragraph_count':len(new)-1,'preserved_original_paragraphs':unchanged,
       'words':len(' '.join(text(n) for n in new).split()),
       'manuscript_status':'non publié',
       'result_claim':'8,27 % lower retrospective pooled MAE versus Chronos-2 stage, four markets, 350 days'}
(ROOT/'tmp/cv_nyx_20260916/edit_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(audit,indent=2,ensure_ascii=True)); print(OUT)
