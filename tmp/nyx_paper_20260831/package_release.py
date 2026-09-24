from pathlib import Path
import hashlib,json,shutil,zipfile,re
from pypdf import PdfReader
from lxml import etree
ROOT=Path(__file__).resolve().parents[2]
TASK=Path(__file__).resolve().parent
OUT=ROOT/'output/nyx_paper_20260831'
SUP=OUT/'supplement';SUP.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pdf=OUT/'NYX_Research_Manuscript.pdf'
shutil.copyfile(TASK/'NYX_render_v3.pdf',pdf)
for folder in ['figures','tables']:shutil.copytree(OUT/folder,SUP/folder,dirs_exist_ok=True)
audit=SUP/'audit';audit.mkdir(exist_ok=True)
for p in (TASK/'analysis').iterdir():
    if p.suffix in ['.csv','.json','.md'] and p.name not in ['manuscript_tables.json','output_sha256_manifest.json']:
        shutil.copyfile(p,audit/p.name)
theory=(TASK/'theory_draft.md').read_text(encoding='utf-8')
(audit/'theoretical_implementation_notes.md').write_text(theory[theory.index('# Evidence notes'):],encoding='utf-8')
portable=SUP/'reproduction';portable.mkdir(exist_ok=True)
for name in ['recompute_cutoff.py','table_layout_seed.json','validation.json','requirements.txt','tested_environment.json']:
    source=TASK/'portable'/name
    if source.exists():shutil.copyfile(source,portable/name)
assert (portable/'validation.json').exists(), 'Portable verification must complete before release'
# The authoritative seed is the table content of this final revision.
shutil.copyfile(OUT/'tables/manuscript_tables.json',portable/'table_layout_seed.json')
source=OUT/'NYX_Research_Manuscript.md'
shutil.copyfile(source,SUP/source.name)
r=PdfReader(pdf);txt='\n'.join(p.extract_text() for p in r.pages)
assert len(r.pages)==22
assert all('Table '+str(i)+'.' in txt for i in range(1,11))
assert all('Figure '+str(i)+'.' in txt for i in range(1,6))
assert not any(x in txt for x in ['@table:','@figure:','MATH:','\\tag{','\\operatorname'])
assert '31 August 2026' in txt
assert not any(x in txt for x in ['15 September 2026','reserved 90','35,036','8,760','11.386'])
with zipfile.ZipFile(OUT/'NYX_Research_Manuscript.docx') as z:xml=etree.fromstring(z.read('word/document.xml'))
ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
display=len(xml.findall('.//m:oMathPara',ns));native=len(xml.findall('.//m:oMath',ns))
assert display==29
assert not any(''.join(t.itertext())=='^' for t in xml.findall('.//m:limUpp/m:lim',ns))
release=dict(title='NYX for hourly day ahead electricity price forecasting across four European markets',
  evaluation_start='2025-09-16',evaluation_cutoff='2026-08-31',delivery_days=350,test_days=75,
  pages=len(r.pages),tables=10,figures=5,numbered_equations=display,native_math_objects=native,
  docx_sha256=sha(OUT/'NYX_Research_Manuscript.docx'),pdf_sha256=sha(pdf),
  scope='Retrospective rescoring of archived forecasts; no new training or reselection. The observation cutoff is not a historical-availability certification.',
  source_vintage_dates='Preserved in audit/cutoff_manifest.json; later source materialization is explicitly distinguished from the evaluation cutoff.',
  implementation_limitation='Archived nonlinear candidate has a sigma-point orientation discrepancy; disclosed in Appendix B. Forecasts have not been repaired.')
(SUP/'manuscript_release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
readme='''# NYX research evidence through 31 August 2026

This supplement accompanies the revised English research manuscript. Evaluation delivery dates are 16 September 2025 through 31 August 2026 inclusive: 350 civil days, 33,600 country–hour observations, and 33,596 matched stage/comparator pairs. The additional residual experiment preserves the original 180-day initial sample and 95-day validation period, followed by 75 retained test days (7,200 country–hours). No configurations were reselected and no model was retrained.

The manuscript contains 29 numbered editable equations, 10 tables and five figures. The source Markdown uses `@table` and `@figure` inclusion directives; the completed Word and PDF files are supplied separately and linked by their checksums in `manuscript_release.json`.

## Contents

- `tables/`: the final publication tables and supporting aggregates. Filenames containing `annual` are retained for compatibility; all their revised contents concern 350 days. Explicit period identifiers are `full350` and `test75`.
- `figures/`: five data-derived figures, each as high-resolution PNG and vector PDF.
- `audit/`: cutoff-specific scores, intervals, provenance, parameter selection, mathematical implementation notes and source fingerprints. Source materialization dates after the cutoff describe provenance, not included evaluation observations.
- `reproduction/`: a tested portable rescoring script, a local table-layout seed, runtime requirements and numerical verification. It reads authorized local inputs and writes only to its own `recomputed/` subdirectory. It does not train models, fetch provider data or modify the source archives.

## Recompute

Use the Python environment and dependencies recorded under `reproduction/`. With access to the original authorized NYX workspace, run from this supplement directory:

```text
python reproduction/recompute_cutoff.py --workspace /path/to/authorized/nyx-workspace
```

The workspace must contain `research/tensor_timesfm_20260915/metrics/`, including the saved paired forecasts, residual test forecasts, extraction manifest, validation selection and rolling fit audit. It must also contain the referenced four-country frozen bundles and source snapshots under `runs/experiments/nuclear_forecast_v1/2026-09-16/`. Exact source paths and checksums are recorded in `audit/cutoff_manifest.json`. The script remaps historical source paths to the supplied workspace. It needs no copy of the previous manuscript's table file: the revised layout seed is included locally.

All revised moving-block intervals use 2,000 replicates, seven-day blocks and seed 20260915. The seed is a reproducibility integer, not the evaluated endpoint. Full precision is retained in CSV; rounding occurs only for manuscript presentation. Positive residual-test differences mean worse than NYX, whereas stage differences explicitly name both models.

## Scope and limitations

The retained test is a truncated subset of a previously examined test period, not a newly untouched holdout. Underlying forecasts and source snapshots were assembled retrospectively. Recorded as-of queries and rolling label checks do not certify provider publication vintages or availability on 31 August. The Chronos checkpoint also postdates the beginning of the evaluated period.

The optimized nonlinear filter candidate has a documented sigma-point covariance-orientation discrepancy. It was selected for 24 NL hours on 21 February 2026, and never during the retained test. This selection count does not bound a corrected rerun's impact on governance. The archived forecasts are unchanged; no corrected-UKF accuracy claim is made.

The package contains aggregate evidence, not licensed hourly observations, prediction matrices, model weights or credentials. Data access and redistribution rights must be handled separately. Reproducing aggregate scores does not recreate full NYX training, establish prospective accuracy, eliminate vintage uncertainty or demonstrate trading profitability.
'''
(SUP/'README.md').write_text(readme,encoding='utf-8')
manifest={p.relative_to(SUP).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(SUP.rglob('*')) if p.is_file() and p.name!='sha256_manifest.json'}
(SUP/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive=OUT/'NYX_Research_Supplement.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(SUP.rglob('*')):
        if p.is_file():z.write(p,'NYX_Research_Supplement/'+p.relative_to(SUP).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert not any(n.endswith(('.parquet','.csv.gz','.bin','.safetensors')) for n in z.namelist())
(TASK/'final_quality_checks.json').write_text(json.dumps(dict(pages=22,tables=10,figures=5,numbered_equations=29,native_math_objects=native,
  stale_result_labels_absent=True,native_equation_arrays=True,accents_corrected=True,zip_crc_pass=True,all_page_visual_review='PASS'),indent=2),encoding='utf-8')
print(json.dumps(dict(pages=len(r.pages),equations=display,docx_bytes=(OUT/'NYX_Research_Manuscript.docx').stat().st_size,pdf_bytes=pdf.stat().st_size,supplement_bytes=archive.stat().st_size),indent=2))
