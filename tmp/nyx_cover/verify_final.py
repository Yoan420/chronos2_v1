from pathlib import Path
from lxml import etree as E
import zipfile, hashlib, json, re
from pypdf import PdfReader

root=Path('C:/Users/BQ6757/chronos2_v1')
src=root/'output/nyx_paper_20250831_20260831/NYX_Research_Manuscript.docx'
out=root/'output/nyx_paper_with_cover/NYX_Research_Manuscript_with_cover.docx'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def load(p):
    with zipfile.ZipFile(p) as z: files={n:z.read(n) for n in z.namelist()}
    return files,E.fromstring(files['word/document.xml'])
a,at=load(src); b,bt=load(out)
ab=at.find('w:body',ns); bb=bt.find('w:body',ns)
def txt(n): return ''.join(n.xpath('.//w:t/text() | .//m:t/text()',namespaces=ns))
def paragraphs(n): return [txt(x) for x in n.findall('.//w:p',ns)]
title=txt(ab[0])
start=next(i for i,n in enumerate(bb) if n.tag=='{'+ns['w']+'}p' and txt(n)==title)
suffix=E.Element('body')
for x in list(bb)[start:-1]: suffix.append(E.fromstring(E.tostring(x)))
source=E.Element('body')
for x in list(ab)[:-1]: source.append(E.fromstring(E.tostring(x)))
ap=paragraphs(source); bp=paragraphs(suffix)
if ap!=bp:
    import difflib
    (root/'tmp/nyx_cover/body_diff.txt').write_text('\n'.join(difflib.unified_diff(ap,bp)),encoding='utf-8')
assert ap==bp,'Body text changed; see body_diff.txt'
am=[txt(x) for x in source.findall('.//m:oMath',ns)]
bm=[txt(x) for x in suffix.findall('.//m:oMath',ns)]
assert am==bm,'Mathematical text changed'
media_src={hashlib.sha256(v).hexdigest() for n,v in a.items() if n.startswith('word/media/')}
media_out={hashlib.sha256(v).hexdigest() for n,v in b.items() if n.startswith('word/media/')}
assert media_src <= media_out,'An original media asset was modified'
pdf=PdfReader(out.with_suffix('.pdf'))
pages=[p.extract_text() for p in pdf.pages]
toc=pages[1]
assert 'Contents' in toc and 'Appendix B' in toc
assert 'Contents will be updated' not in '\n'.join(pages)
normal=lambda t: re.sub(r'\s+',' ',t).strip()
assert normal(title) in normal(pages[2]), 'Body not starting on third page'
toc_checks=[]
for p in bt.findall('.//w:p',ns):
    style=p.find('w:pPr/w:pStyle',ns)
    if style is not None and style.get('{'+ns['w']+'}val') in ('TOC1','TOC2'):
        runs=p.xpath('.//w:t/text()',namespaces=ns)
        label=normal(''.join(runs[:-1])); page=int(runs[-1])
        assert label in normal(pages[page+1]), f'TOC page incorrect: {label} -> {page}'
        toc_checks.append({'heading':label,'page':page})
assert len(toc_checks)==23, f'Unexpected TOC entries: {len(toc_checks)}'
audit={'source_hash':hashlib.sha256(src.read_bytes()).hexdigest(),
       'body_paragraphs_preserved':len(ap),'math_objects_preserved':len(am),
       'tables_source':len(source.findall('.//w:tbl',ns)),
       'tables_body_output':len(suffix.findall('.//w:tbl',ns)),
       'all_original_media_preserved':True,'sections':len(bt.findall('.//w:sectPr',ns)),
       'pdf_pages':len(pages),'contents_text':toc,
       'first_body_page_tail':pages[2][-100:],'toc_verified_entries':toc_checks}
(root/'tmp/nyx_cover/final_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(audit,indent=2,ensure_ascii=False))
