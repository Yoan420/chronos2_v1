from pathlib import Path
import zipfile,hashlib,json
from lxml import etree
ROOT=Path(__file__).resolve().parents[2]
TASK=Path(__file__).resolve().parent
SOURCE=ROOT/'output/nyx_paper_20260831/NYX_Research_Manuscript.docx'
OUT=ROOT/'output/nyx_paper_20250831_20260831'
DEST=OUT/'NYX_Research_Manuscript.docx'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(SOURCE) as z:parts={n:z.read(n) for n in z.namelist()}
xml=etree.fromstring(parts['word/document.xml'])
paragraphs=xml.findall('.//w:body/w:p',ns)
get=lambda p:''.join(p.xpath('.//w:t/text()',namespaces=ns))
changes=[]
def replace_par(prefix,replacement):
    matches=[p for p in paragraphs if get(p).startswith(prefix)]
    assert len(matches)==1,(prefix,len(matches))
    p=matches[0];old=get(p);new=replacement(old) if callable(replacement) else replacement
    assert p.find('.//m:oMath',ns) is None
    nodes=p.findall('.//w:t',ns);assert nodes
    nodes[0].text=new;nodes[0].set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    for t in nodes[1:]:t.text=''
    changes.append({'old':old,'new':new})

replace_par('Research manuscript', 'Research manuscript · Study period 31 August 2025 to 31 August 2026')
old_sentence='A retrospective prequential evaluation covers Belgium, Germany, France and the Netherlands from 16 September 2025 to 31 August 2026.'
new_sentence='The study period is 31 August 2025 to 31 August 2026. Available archived NYX forecasts support a retrospective prequential evaluation of Belgium, Germany, France and the Netherlands from 16 September 2025 to 31 August 2026.'
def abstract(text):
    assert text.count(old_sentence)==1
    return text.replace(old_sentence,new_sentence)
replace_par('Pretrained time series models',abstract)
replace_par('The target is the hourly day-ahead electricity price', 'The target is the hourly day-ahead electricity price, in EUR/MWh, for Belgium (BE), Germany (DE), France (FR) and the Netherlands (NL). The study period runs from 31 August 2025 through 31 August 2026 inclusive, a window of 366 civil days. Complete final NYX forecasts in the retained four-country archive begin on 16 September 2025. The first 16 days of the study window are therefore not scored; they are neither filled nor represented by the reported results. The evaluated sample remains 350 civil days, with 8,400 physical hourly observations per country and 33,600 country–hour pairs in total. It contains one 23-hour day, one 25-hour day and 348 ordinary days. Every evaluated delivery date is no later than 31 August 2026. The study window does not assert that the retrospectively assembled forecast and observation snapshots were already available at their historical forecast origins.')

# Keep the user's current Word package, including title, deletions, style changes,
# native equations and figures. Only the three identified prose paragraphs change.
parts['word/document.xml']=etree.tostring(xml,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
before=etree.fromstring(zipfile.ZipFile(SOURCE).read('word/document.xml'))
count=lambda tree,key:len(tree.findall(key,ns))
for key in ['.//m:oMath','.//m:oMathPara','.//w:tbl','.//w:drawing']:
    assert count(xml,key)==count(before,key),(key,count(xml,key),count(before,key))
assert get(paragraphs[0])=='NYX for hourly day ahead electricity price forecasting across four CWE'
assert not any(get(p).startswith('Correspondence:') for p in paragraphs)
assert not any(get(p)=='Data and code availability' for p in paragraphs)
sourcehash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
summary={'requested_study_period':['2025-08-31','2026-08-31'],'study_civil_days_inclusive':366,
  'verified_evaluation_period':['2025-09-16','2026-08-31'],'evaluated_civil_days':350,
  'unscored_initial_days':16,'original_numerical_results_unchanged':True,
  'user_edits_preserved':True,'changed_paragraphs':3,'source_docx_sha256':sourcehash,
  'docx_sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'changes':changes}
(TASK/'revision_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='changes'},indent=2))
