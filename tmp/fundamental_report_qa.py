"""Read-only QA of completed frozen reports; only writes generated validation JSON."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from economic_value.data import _plots
from marginal_cost_expert.evaluation import _array
from nyx_scarcity.reporting import _prepare
from nyx_scarcity_zonal.reporting import _verified_storm
from nyx_fundamental_stress.runner import read_suite, verify_result, protected_state
from nyx_fundamental_stress.reporting import _validate_decisions

NODE=Path(r'C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-10,equal_nan=True)

def main():
    directory=Path(sys.argv[1]).resolve()
    status=json.loads((directory/'status.json').read_text(encoding='utf8'))
    assert status['status']=='completed',status
    directory,config,manifest=read_suite(directory,root=ROOT)
    for variant in config['variants']:verify_result(directory,variant,manifest)
    before=protected_state(ROOT)
    watched=[*directory.glob('reports/*.html'),*directory.glob('reports_fixed25/*.html'),directory/'source_audit.json',
             directory/'fundamental/predictions.parquet',directory/'fundamental/proposals_25.parquet']
    initial={str(path):sha(path) for path in watched}
    source=json.loads((directory/'source_audit.json').read_text(encoding='utf8'))
    combined={}
    bundles=[]
    for folder,filename,policy in [('reports','predictions.parquet','governed'),('reports_fixed25','proposals_25.parquet','fixed25')]:
        frame=_prepare(pd.read_parquet(directory/'fundamental'/filename))
        audit=json.loads((directory/folder/'fundamental_report_audit.json').read_text(encoding='utf8'))
        checks={}
        for zone in ['FR','DE','BE','NL']:
            group=frame.loc[frame.zone.eq(zone)].reset_index(drop=True)
            evaluation=group.loc[group['sample'].eq('evaluation')]
            live=group.loc[group['sample'].eq('live')]
            proof=audit['reports'][zone]
            assert len(evaluation)==8760 and evaluation.local_day.nunique()==365 and len(live)==24
            assert proof['paired_hours']==8735 and proof['live_excluded_from_statistics'] is True
            assert proof['evaluation_start_day']=='2025-09-15' and proof['evaluation_end_day']=='2026-09-14'
            assert proof['strict_governor_enforced']==(policy=='governed')
            assert proof['decision_checks']['saved_physical_gate_checked'] is True
            assert proof['decision_checks']['saved_probability_gate_checked'] is True
            assert all(c.startswith('feature_fundamental_') for c in proof['allowed_input_features'])
            checked=_validate_decisions(group,policy)
            storm,_=_verified_storm(group,source,zone)
            assert storm['panel_missing_values_filled'] is False
            close(group.candidate_forecast,group.forecast+group.applied_correction)
            close(group.applied_correction,group.selected_weight*group.bounded_correction)
            path=next((directory/folder).glob(f'forecast_{zone.lower()}_*.html'))
            text=path.read_text(encoding='utf8')
            for word in ['EXPERT FONDAMENTAL','STATISTICS — PRIX MOYENS','storm-comparison','hourly-comparison',
                         'Motif de proposition','Médiane d’erreur signée','prévalence réelle','28 jours de calibration','90 jours' if policy=='governed' else 'poids fixe de 25 %']:
                assert word in text,(zone,word)
            assert 'gouverneur annuel strict' not in text and 'p &gt; 0,6' not in text
            for feature in [c for c in group if c.startswith('feature_') and not c.startswith('feature_fundamental_')]:
                assert feature not in text,(zone,feature)
            scripts=re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>',text,re.S)
            compiled=subprocess.run([str(NODE),'--check'],input='\n'.join(scripts),encoding='utf8',text=True,capture_output=True)
            assert compiled.returncode==0,compiled.stderr
            statistics=next(s for s in scripts if 'const payload =' in s and 'statistics-metric-select' in s)
            payload=json.JSONDecoder().raw_decode(statistics.split('const payload =',1)[1].lstrip())[0]
            daily=[r for r in payload['records'] if r['sample']=='daily']
            assert len(daily)==365 and max(r['period_key'] for r in daily)=='2026-09-14'
            assert min(r['period_key'] for r in daily)=='2025-09-15'
            assert next(r for r in daily if r['period_key']=='2025-09-15')['benchmark_mean_price'] is None
            assert next(r for r in daily if r['period_key']=='2025-10-26')['benchmark_n']==24
            means={}
            for day in ['2026-09-14','2026-06-24','2026-06-25','2026-06-26']:
                row=next(r for r in daily if r['period_key']==day)
                block=evaluation.loc[evaluation.local_day.eq(day)]
                valid=np.isfinite(block[['actual','forecast','candidate_forecast','benchmark_forecast']]).all(axis=1)
                block=block.loc[valid]
                for key,column in [('mean_price','candidate_forecast'),('observed_mean_price','actual'),('benchmark_mean_price','benchmark_forecast')]:
                    close(row[key],block[column].mean())
                means[day]={key:row[key] for key in ['mean_price','observed_mean_price','benchmark_mean_price']}
            # Every saved live quantile must be present unchanged in an actual plot trace.
            traces=[trace for plot in _plots(text) for trace in plot]
            live_traces=[t for t in traces if len(t.get('x',[]))==24 and str(t.get('x',[''])[0]).startswith('2026-09-15')]
            for column in ['candidate_forecast','candidate_q10','candidate_q90']:
                assert any(np.allclose(_array(t.get('y',[])),live[column],atol=1e-8,rtol=1e-10,equal_nan=True)
                           for t in live_traces if len(_array(t.get('y',[])))==24),(zone,column)
            banner=text.split('data-report-section="fundamental-experiment"',1)[1].split('</section>',1)[0]
            table_rows=re.findall(r'<tr>(.*?)</tr>',banner,re.S)[1:]
            assert len(table_rows)==24
            for saved,row in zip(live.to_dict('records'),table_rows):
                cells=re.findall(r'<td>(.*?)</td>',row,re.S)
                close(float(cells[2]),round(saved['predicted_signed_residual_median'],3))
                assert cells[5]==('Oui' if saved['physical_gate_passed'] else 'Non')
                assert saved['proposal_reason'] in cells[6]
                close(float(cells[8]),round(saved['selected_weight'],3))
            bundles.append({'zone':zone,'policy':policy,'statistics':statistics,
                'bootstrap':next(s for s in scripts if 'document.documentElement.dataset.theme = theme' in s),
                'theme':next(s for s in scripts if 'function applyPlotlyTheme' in s)})
            checks[zone]={'statistics_days':365,'paired_hours':8735,'statistics_end':'2026-09-14',
                'live_day':'2026-09-15','live_point_q10_q90_verified':True,'means':means,
                'gate_checks':checked,'storm_original_archive_verified':True,'frozen_missing_values_preserved':True,
                'actual_diagnostic_columns_verified':True,'file':str(path)}
        index=(directory/folder/'index.html').read_text(encoding='utf8')
        assert '../fundamental_comparison.html' in index and 'Corrections aggravantes' in index
        combined[folder]={'status':'passed','variant':'fundamental','decision_policy':policy,'checks':checks,
            'index_comparison_link_and_harm_table':True,'source_result_verified':True,'no_render_or_fit_performed':True}
    js=subprocess.run([str(NODE),str(ROOT/'tmp/fundamental_report_qa.js')],input=json.dumps(bundles),
                      encoding='utf8',text=True,capture_output=True)
    assert js.returncode==0,js.stderr
    results=json.loads(js.stdout)
    assert initial=={str(path):sha(path) for path in watched} and before==protected_state(ROOT)
    for folder,result in combined.items():
        result.update(javascript=[r for r in results if r['policy']==result['decision_policy']],
            html_predictions_code_and_protected_files_unchanged_during_qa=True)
        (directory/folder/'report_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'status':'passed','reports':8,'javascript_contexts':len(results),'snapshot':str(directory)},ensure_ascii=False))

if __name__=='__main__':main()
