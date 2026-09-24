from pathlib import Path
import hashlib,json,shutil,zipfile
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2]
TASK=Path(__file__).resolve().parent
OUT=ROOT/'output/nyx_paper_20260916'
pdf=OUT/'NYX_Research_Manuscript.pdf'
shutil.copyfile(TASK/'NYX_render_final.pdf',pdf)
sup=OUT/'supplement'
shutil.copytree(OUT/'figures',sup/'figures',dirs_exist_ok=True)
shutil.copytree(OUT/'tables',sup/'tables',dirs_exist_ok=True)
r=PdfReader(pdf)
assert len(r.pages)==18
text='\n'.join(p.extract_text() for p in r.pages)
assert all('Table '+str(i)+'.' in text for i in range(1,11))
assert all('Figure '+str(i)+'.' in text for i in range(1,6))
assert not any(x in text for x in ['@table:','@figure:','MATH:','\\tag{'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary={'manuscript_title':'NYX for hourly day ahead electricity price forecasting across four European markets','date':'2026-09-16','pages':len(r.pages),'tables':10,'figures':5,'docx_sha256':sha(OUT/'NYX_Research_Manuscript.docx'),'pdf_sha256':sha(pdf),'format':'Single-column research manuscript with editable Word equations and vector figure assets','scope':'Retrospective empirical study of retained NYX forecasts, not a prospective deployment validation'}
(sup/'manuscript_release.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
readme=(sup/'README.md').read_text(encoding='utf-8')
readme += '\n## Manuscript assets\n\nThe final manuscript contains 18 pages, 10 tables and five figures. The `figures/` directory contains the five data-derived figures in high-resolution PNG and vector PDF formats. The `tables/manuscript_tables.json` file records the exact formatted table content. `manuscript_release.json` links this evidence package to the final Word and PDF files by SHA256; those two manuscript files are delivered separately. Figure assets are not additional experimental runs.\n'
(sup/'README.md').write_text(readme,encoding='utf-8')
manifest={str(p.relative_to(sup)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(sup.rglob('*')) if p.is_file() and p.name!='sha256_manifest.json'}
(sup/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive=OUT/'NYX_Research_Supplement.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(sup.rglob('*')):
        if p.is_file():z.write(p,'NYX_Research_Supplement/'+p.relative_to(sup).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert not any(n.endswith(('.parquet','.csv.gz','.bin','.safetensors')) for n in z.namelist())
(TASK/'final_quality_checks.json').write_text(json.dumps({'pages':18,'table_labels_present':10,'figure_labels_present':5,'raw_directives_absent':True,'docx_native_math':20,'zip_crc_pass':True,'supplement_files':len(manifest)+1,'main_visual_review_pages_1_to_9':'PASS','other_visual_review':'separate agent confirmation'},indent=2),encoding='utf-8')
print(json.dumps({'pages':len(r.pages),'pdf_bytes':pdf.stat().st_size,'docx_bytes':(OUT/'NYX_Research_Manuscript.docx').stat().st_size,'supplement_bytes':archive.stat().st_size,'supplement_files':len(manifest)+1},indent=2))
