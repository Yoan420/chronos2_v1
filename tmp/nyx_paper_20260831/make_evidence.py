from pathlib import Path
import sys, os, json, hashlib, shutil
ROOT=Path(__file__).resolve().parents[2]
TASK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp/nyx_paper_20260916/deps'))
os.environ['MPLCONFIGDIR']=str(TASK/'mplconfig')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'output/nyx_paper_20260831'
FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
TAB=OUT/'tables'; TAB.mkdir(exist_ok=True)
SRC=TASK/'analysis'
f=pd.read_csv(SRC/'verified_hourly_pairs.csv.gz')
f=f[f.day<='2026-08-31'].copy()
assert f.day.max()=='2026-08-31' and len(f)==33600
f['date']=pd.to_datetime(f.day)
m=json.loads((SRC/'extraction_manifest.json').read_text())
zones=['BE','DE','FR','NL']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelsize':10,'legend.frameon':False,'savefig.dpi':240,'axes.grid':False})
colors={'chronos':'#8a8a8a','residual':'#cb7f32','nyx':'#175f82','storm':'#7f4b75'}
def save(name):
    plt.savefig(FIG/(name+'.png'),dpi=240,bbox_inches='tight',facecolor='white')
    plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close()
def met(a,col):
    e=a[col]-a.observed
    return [len(e),abs(e).mean(),np.sqrt((e*e).mean()),e.mean()]
rows=[]
for z in zones:
    g=f[f.zone==z].dropna(subset=['chronos','residual','nyx','storm','observed'])
    for col in colors: rows.append([z,col,*met(g,col)])
scores=pd.DataFrame(rows,columns=['zone','model','n','mae','rmse','bias'])
scores.to_csv(TAB/'annual_paired_metrics.csv',index=False)
stats=[]
for z in zones:
    y=f.loc[f.zone==z,'observed']
    stats.append([z,len(y),y.mean(),y.std(),y.min(),y.median(),y.max(),int((y<0).sum())])
pd.DataFrame(stats,columns=['zone','n','mean','sd','min','median','max','negative_hours']).to_csv(TAB/'data_description.csv',index=False)
reg=[]
for z in zones:
    g=f[f.zone==z]; threshold=m['thresholds_initial_train_frozen_actual'][z]['q99']
    for name,gg in [('negative',g[g.observed<0]),('high',g[g.observed>threshold])]:
        reg.append([z,name,threshold,*met(gg,'nyx'),100*((gg.observed>=gg.nyx_p10)&(gg.observed<=gg.nyx_p90)).mean()])
pd.DataFrame(reg,columns=['zone','regime','q99_initial','n','mae','rmse','bias','coverage80_pct']).to_csv(TAB/'extreme_regimes.csv',index=False)
for name in ['locked_test_metrics.csv','locked_test_paired_bootstrap.csv','nyx_interval_calibration.csv','correction_saturation.csv']:
    shutil.copyfile(SRC/name,TAB/name)

# Fig 1: architecture and the separately assessed residual experiment.
fig,ax=plt.subplots(figsize=(7.4,3.9)); ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
items=[(1.5,4.1,'Prices and calendar\nFive residual-load forecasts\nFrench nuclear forecast'),(5,4.1,'Frozen Chronos-2\n2,048-hour context'),(8.5,4.1,'Quantiles\nP10  /  P50  /  P90'),(8.5,2.6,'CatBoost residual shift\nDaily 365-day refit'),(5,2.6,'Governed filter shift\nDaily 365-day replay'),(1.5,2.6,'NYX quantiles\nWidths unchanged')]
for x,y,label in items: ax.text(x,y,label,ha='center',va='center',fontsize=9.3,bbox=dict(boxstyle='round,pad=.5',fc='#f3f5f6',ec='#888888',lw=.7))
for start,end in [((3.18,4.1),(3.8,4.1)),((6.2,4.1),(7.5,4.1)),((8.5,3.62),(8.5,3.05)),((7.15,2.6),(6.5,2.6)),((3.55,2.6),(2.6,2.6))]: ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#444444',lw=1.2))
ax.text(5,1.58,'ADDITIONAL RESIDUAL EXPERIMENT',ha='center',weight='bold',fontsize=9.5)
for x,w,c,label in [(0,180/35,'#c8d4da','Initial 180 days'),(180/35,95/35,'#dbe2e6','Validation 95 days'),(275/35,75/35,'#eceff1','Reserved 75 days')]:
    ax.barh(.95,w,left=x,height=.48,color=c,edgecolor='white');ax.text(x+w/2,.95,label,ha='center',va='center',fontsize=9)
ax.text(0,.35,'16 Sep 2025',ha='left',fontsize=8.5); ax.text(180/35,.35,'15 Mar 2026',ha='center',fontsize=8.5);ax.text(275/35,.35,'18 Jun 2026',ha='center',fontsize=8.5);ax.text(10,.35,'31 Aug 2026',ha='right',fontsize=8.5)
save('figure1_design')

# Fig 2: month-by-month performance on exactly common hours.
fig,axes=plt.subplots(2,2,figsize=(7.4,5),sharex=True,sharey=True)
monthly=[]
for z,ax in zip(zones,axes.flat):
    g=f[f.zone==z].dropna(subset=['chronos','nyx','storm']).copy();g['ym']=g.date.dt.to_period('M').astype(str)
    for col in ['chronos','nyx','storm']:
        a=g.assign(ae=abs(g[col]-g.observed)).groupby('ym').ae.mean()
        monthly.extend([[z,col,k,v] for k,v in a.items()])
        ax.plot(range(len(a)),a.values,label={'chronos':'Chronos-2','nyx':'NYX','storm':'Storm composite'}[col],color=colors[col],lw=1.6,marker='o',ms=2.5)
    ax.set_title(z,loc='left',fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_xticks([0,3,6,9,11],['Sep 25','Dec','Mar 26','Jun','Aug']);ax.tick_params(axis='x',labelsize=8)
for ax in axes[:,0]:ax.set_ylabel('MAE (EUR/MWh)')
axes[0,0].legend(fontsize=8,loc='upper left');fig.tight_layout()
save('figure2_monthly_mae')
pd.DataFrame(monthly,columns=['zone','model','month','mae']).to_csv(TAB/'monthly_mae.csv',index=False)

# Fig 3: interval coverage by diagnostically defined realized regime.
cal=pd.read_csv(SRC/'nyx_interval_calibration.csv')
fig,ax=plt.subplots(figsize=(7.4,3.35)); x=np.arange(4);w=.23
for j,(label,sel,color) in enumerate([('All hours','all','#175f82'),('Negative prices','negative','#7a9ead'),('Prices above initial q99','high','#cb7f32')]):
    vals=[]
    for z in zones:
        g=f[f.zone==z]
        if sel=='negative':g=g[g.observed<0]
        elif sel=='high':g=g[g.observed>m['thresholds_initial_train_frozen_actual'][z]['q99']]
        vals.append(100*((g.observed>=g.nyx_p10)&(g.observed<=g.nyx_p90)).mean())
    bars=ax.bar(x+(j-1)*w,vals,w,label=label,color=color)
    ax.bar_label(bars,labels=[f'{v:.1f}' for v in vals],fontsize=8,padding=2)
ax.axhline(80,color='black',ls='--',lw=1,label='Nominal 80%');ax.set_ylim(0,101);ax.set_xticks(x,zones);ax.set_ylabel('P10–P90 coverage (%)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=8);fig.tight_layout()
save('figure3_interval_coverage')

# Fig 4: simultaneous residual correlation is not a forecast gain.
train=f[f.split=='train'].pivot(index='timestamp',columns='zone',values='residual_nyx')[zones]
corr=train.corr(); cov=np.cov(train.to_numpy(),rowvar=False);ev=np.linalg.eigvalsh(cov)[::-1];shares=ev/ev.sum()
fig,(ax,ax2)=plt.subplots(1,2,figsize=(7.4,3.3),gridspec_kw={'width_ratios':[1,1]})
im=ax.imshow(corr,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),zones);ax.set_yticks(range(4),zones);ax.set_title('(a) Residual correlations',fontsize=10)
for i in range(4):
    for j in range(4):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if corr.iloc[i,j]>.65 else 'black')
ax2.bar(range(1,5),100*shares,color='#175f82');ax2.plot(range(1,5),100*np.cumsum(shares),'o-',color='#cb7f32',lw=1.3,label='Cumulative');ax2.set_xticks(range(1,5));ax2.set_xlabel('Principal component');ax2.set_ylabel('Explained variance (%)');ax2.set_ylim(0,108);ax2.set_title('(b) Covariance eigenvalues',fontsize=10);ax2.legend(fontsize=8);fig.tight_layout()
save('figure4_residual_dependence')

# Fig 5: point estimates and dependence-preserving uncertainty.
b=pd.read_csv(SRC/'locked_test_paired_bootstrap.csv'); b=b[(b.label=='latest')&(b.zone=='pooled')&(b.model!='nyx')].copy()
names={'ewma_country':'EWMA country','ewma_country_hour':'EWMA country and hour','ridge_univariate':'Univariate ridge','ridge_multivariate':'Multivariate ridge','pca_ridge':'PCA ridge','storm':'Storm composite'}
fig,ax=plt.subplots(figsize=(7.4,3.1));yy=np.arange(len(b))
ax.errorbar(b.mae_delta,yy,xerr=np.vstack([b.mae_delta-b.mae_delta_ci_low,b.mae_delta_ci_high-b.mae_delta]),fmt='o',color='#175f82',ecolor='#175f82',capsize=3)
ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(yy,[names.get(v,v) for v in b.model]);ax.invert_yaxis();ax.set_xlabel('MAE difference from NYX (EUR/MWh)');ax.grid(axis='x',alpha=.2);fig.tight_layout()
save('figure5_test_uncertainty')
manifest={'source_matrix_sha256':hashlib.sha256((SRC/'verified_hourly_pairs.csv.gz').read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((SRC/'extraction_manifest.json').read_bytes()).hexdigest(),'rows':len(f),'countries':zones,'evaluation_cutoff':'2026-08-31','source_artifact_date':'2026-09-16','matplotlib':matplotlib.__version__,'python':sys.version,'figures':[p.name for p in FIG.glob('*.png')]}
(OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(scores.round(4).to_string(index=False));print('Figure models',b.model.tolist());print('Created five figures and audited tables.')
