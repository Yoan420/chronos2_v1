from pathlib import Path
from lxml import etree as E
import zipfile

ROOT = Path('C:/Users/BQ6757/chronos2_v1')
SRC = ROOT/'output/cv_soda_nyx_link_20260917/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
OUT = ROOT/'output/cv_soda_sg_networks_20260917/Yoan_Kesraoui_CV_These_SODA_NYX.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
old = 'étude de réseaux neuronaux apprenant conjointement les prix et leurs sensibilités aux paramètres de Heston ; analyse de la précision, de la stabilité et de la généralisation pour la calibration.'
new = 'étude de réseaux denses multicouches (MLP) à apprentissage différentiel (DDN) pour le pricing sous Heston ; apprentissage conjoint des prix et de leurs sensibilités aux paramètres, analyse de la précision, de la stabilité et de la généralisation pour la calibration.'
with zipfile.ZipFile(SRC) as z:
    original = {n: z.read(n) for n in z.namelist()}
root = E.fromstring(original['word/document.xml'])
matches = [t for t in root.findall('.//w:t', NS) if t.text == old]
assert len(matches) == 1, len(matches)
matches[0].text = new
updated = original.copy()
updated['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, value in updated.items():
        z.writestr(name, value)
matches[0].text = old
assert E.tostring(root) == E.tostring(E.fromstring(original['word/document.xml']))
print(OUT)
print('Only the SG research mission was refined; all other text and links preserved.')
