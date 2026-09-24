from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TASK=Path(__file__).resolve().parent
old=ROOT/'tmp/nyx_paper_20260916'
s=(old/'make_evidence.py').read_text(encoding='utf-8')
s=s.replace("str(TASK/'deps')", "str(ROOT/'tmp/nyx_paper_20260916/deps')")
s=s.replace("OUT=ROOT/'output/nyx_paper_20260916'", "OUT=ROOT/'output/nyx_paper_20260831'")
s=s.replace("SRC=ROOT/'tmp/tensor_timesfm_audit/metrics'", "SRC=TASK/'analysis'")
s=s.replace("f['date']=pd.to_datetime(f.day)", "f=f[f.day<='2026-08-31'].copy()\nassert f.day.max()=='2026-08-31' and len(f)==33600\nf['date']=pd.to_datetime(f.day)")
s=s.replace("(0,4.93,'#c8d4da','Initial 180 days'),(4.93,2.6,'#dbe2e6','Validation 95 days'),(7.53,2.47,'#eceff1','Reserved 90 days')", "(0,180/35,'#c8d4da','Initial 180 days'),(180/35,95/35,'#dbe2e6','Validation 95 days'),(275/35,75/35,'#eceff1','Reserved 75 days')")
s=s.replace("ax.text(4.93,.35", "ax.text(180/35,.35").replace("ax.text(7.53,.35", "ax.text(275/35,.35").replace("15 Sep 2026", "31 Aug 2026")
s=s.replace("[0,3,6,9,12],['Sep 25','Dec','Mar 26','Jun','Sep 26']", "[0,3,6,9,11],['Sep 25','Dec','Mar 26','Jun','Aug']")
s=s.replace("'analysis_date':'2026-09-16'", "'evaluation_cutoff':'2026-08-31','source_artifact_date':'2026-09-16'")
(TASK/'make_evidence.py').write_text(s,encoding='utf-8')
s=(old/'build_manuscript.py').read_text(encoding='utf-8')
s=s.replace("OUT=ROOT/'output/nyx_paper_20260916'", "OUT=ROOT/'output/nyx_paper_20260831'")
s=s.replace("SRC=ROOT/'tmp/tensor_timesfm_audit/metrics'", "SRC=TASK/'analysis'")
s=s.replace("TASK/'verified_stage_metrics.csv'", "SRC/'verified_stage_metrics.csv'").replace("TASK/'storm_provenance_sensitivity.csv'", "SRC/'storm_provenance_sensitivity.csv'")
s=s.replace("shutil.copyfile(TASK/name", "shutil.copyfile(SRC/name")
s=s.replace("8,760", "8,400").replace("365 delivery days", "350 delivery days").replace("8,759", "8,399").replace("35,036", "33,596").replace("8,640", "7,200")
s=s.replace("Annual MAE", "Evaluation period MAE").replace("Annual NYX", "Evaluation period NYX")
s=s.replace("reserved 90 days", "reserved 75 days").replace("('annual','Annual'),('test90','Reserved 90 d')", "('annual','350 days'),('test75','Reserved 75 d')")
s=s.replace(",('cache_only_no_label_fallback','Also exclude EPEX label fallback')", "")
s=s.replace("Appendix B specifies filter dynamics.", "Section 2.3 specifies filter dynamics.")
s=s.replace("the final bounded weighted shift is b in Equation (11).", "the final bounded weighted shift is b in Section 2.3.")
s=s.replace("for name,size,space in [('Title',19,10),('Heading 1',13,11),('Heading 2',11.8,9)]", "for name,size,space in [('Title',19,10),('Heading 1',13,11),('Heading 2',11.8,9),('Heading 3',11.3,8)]")
start=s.index('def add_inline(');end=s.index('\ntables={}',start)
s=s[:start]+r'''def math(p,latex,display=False):
    tag=re.search(r'\\tag\{([^}]+)\}',latex)
    latex=re.sub(r'\\tag\{[^}]+\}','',latex).strip()
    # A genuine Word display equation allows full-height fractions and matrices.
    converted=XSLT(etree.fromstring(convert(latex,display='block' if display else 'inline').encode())).getroot()
    if converted.tag==qn('m:oMathPara'):
        formula=converted.find(qn('m:oMath'))
    else:formula=converted
    if tag:
        run=OxmlElement('m:r');prop=OxmlElement('m:rPr');style=OxmlElement('m:sty');style.set(qn('m:val'),'p');prop.append(style);run.append(prop)
        t=OxmlElement('m:t');t.set(qn('xml:space'),'preserve');t.text='    ('+tag.group(1)+')';run.append(t);formula.append(run)
    if display:
        container=OxmlElement('m:oMathPara');props=OxmlElement('m:oMathParaPr');jc=OxmlElement('m:jc');jc.set(qn('m:val'),'center');props.append(jc);container.append(props);container.append(formula);p._p.append(container)
    else:p._p.append(formula)

def add_inline(p,text):
    pat=r'\$([^$]+)\$|\[([^\]]+)\]\((https?://[^)]+)\)'
    pos=0
    for m in re.finditer(pat,text):
        p.add_run(text[pos:m.start()])
        if m.group(1) is not None:math(p,m.group(1))
        else:
            h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(m.group(3),RT.HYPERLINK,is_external=True));r=OxmlElement('w:r');pr=OxmlElement('w:rPr');c=OxmlElement('w:color');c.set(qn('w:val'),'000000');pr.append(c);r.append(pr);t=OxmlElement('w:t');t.text=m.group(2);r.append(t);h.append(r);p._p.append(h)
        pos=m.end()
    p.add_run(text[pos:])
''' + s[end:]
s=s.replace("elif block.startswith('### '):doc.add_paragraph(block[4:],style='Heading 2')", "elif block.startswith('### '):doc.add_paragraph(block[4:],style='Heading 2')\n    elif block.startswith('#### '):doc.add_paragraph(block[5:],style='Heading 3')")
s=s.replace("p.paragraph_format.space_before=Pt(3);math(p,block[2:-2])", "p.paragraph_format.space_before=Pt(6);p.paragraph_format.keep_together=True;math(p,block[2:-2],display=True)")
(TASK/'build_manuscript.py').write_text(s,encoding='utf-8')
print('Prepared cutoff builders with native display and inline equations.')
