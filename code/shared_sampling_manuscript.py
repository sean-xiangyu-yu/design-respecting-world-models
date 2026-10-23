"""Reproduce the two shared-sampling scenarios and manuscript figures.

Fit 648 cluster-months. Relative sampling intensity r in {.8,.6,.4} is
calibrated using enrollment totals. Both outcome categories share s_ct.
The existing task generator preserves paired random draws from the earlier
analysis; only the two shared-sampling scenarios are fitted and reported here.
"""
import json, time, platform, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cluster_month_analysis import ROOT, data, fit, tasks, run_replicate, KINDS, COLORS
OUT=ROOT/'outputs/cluster_month_shared_sampling'
OUT.mkdir(parents=True,exist_ok=True)
SCENARIOS=('shared_monthly','shared_cluster_monthly')
R=(.8,.6,.4)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})

def shared_tasks(y,n,z,factual):
    for task in tasks(y,n,z,factual,500):
        task=list(task)
        task[6]=[(r,[s for s in scenarios if s[0] in SCENARIOS]) for r,scenarios in task[6]]
        yield tuple(task)

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

def sampling_table(summary):
    lines=[r'\begin{table}[ht]',r'\centering\small',
           r'\caption{Median paired ratio changes (percent), with central 95\% ranges over 500 replicates. Both labels share a sampling probability within each cluster-month.}',
           r'\label{tab:sampling}',r'\begin{tabular}{llrr}',r'\toprule',
           r'Policy & Intensity & Design-ignoring & Design-respecting\\',r'\midrule']
    for scenario,label in zip(SCENARIOS,['Monthly','Cluster-month']):
        for r in R:
            cells=[]
            for model in KINDS:
                row=summary.loc[(summary.scenario==scenario)&(summary.relative_sampling_intensity==r)&(summary.model==model)].iloc[0]
                cells.append(f'${row["median"]:.2f}$ [$ {row.q025:.2f}, {row.q975:.2f} $]')
            lines.append(f'{label} & {r*100:.0f}'+r'\% & '+' & '.join(cells)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (OUT/'sampling_table.tex').write_text('\n'.join(lines)+'\n')

def sampling_figures(draws):
    for scenario,name in [('shared_cluster_monthly','cluster_month_sampling')]:
        fig,ax=plt.subplots(figsize=(3.5,3.2))
        for j,r in enumerate(R):
            for k,off in zip(KINDS,[-.16,.16]):
                v=draws.loc[(draws.scenario==scenario)&(draws.relative_sampling_intensity==r)&(draws.model==k),'relative_change_percent']
                ax.boxplot([v],positions=[j+off],widths=.25,orientation='vertical',
                           whis=(2.5,97.5),showfliers=False,manage_ticks=False,patch_artist=True,
                           boxprops={'facecolor':COLORS[k],'edgecolor':COLORS[k],'alpha':.3},
                           medianprops={'color':COLORS[k],'lw':1.2},
                           whiskerprops={'color':COLORS[k]},capprops={'color':COLORS[k]})
        ax.axhline(0,color='#667085',ls='--',lw=.8)
        ax.set_xticks(range(3),['80%','60%','40%']);ax.set_xlim(-.55,2.55)
        ax.set_ylim(-85,45);ax.set_yticks([-80,-60,-40,-20,0,20,40])
        ax.set_xlabel('Relative sampling intensity',labelpad=5)
        ax.set_ylabel('Paired ratio change (%)',labelpad=5)
        ax.grid(axis='y',alpha=.15)
        fig.legend([Line2D([0],[0],color=COLORS[k],lw=2) for k in KINDS],
                   ['Design-ignoring','Design-respecting'],loc='upper center',
                   ncol=2,frameon=False,bbox_to_anchor=(.55,.99),fontsize=8)
        fig.subplots_adjust(left=.17,right=.98,bottom=.15,top=.87)
        save(fig,name)

    fig,ax=plt.subplots(figsize=(1.9,1.95))
    for j,r in enumerate(R):
        for k,off in zip(KINDS,[-.16,.16]):
            v=draws.loc[(draws.scenario=='shared_monthly')&(draws.relative_sampling_intensity==r)&(draws.model==k),'relative_change_percent']
            ax.boxplot([v],positions=[j+off],widths=.25,whis=(2.5,97.5),showfliers=False,manage_ticks=False,patch_artist=True,
                       boxprops={'facecolor':COLORS[k],'edgecolor':COLORS[k],'alpha':.3},
                       medianprops={'color':COLORS[k],'lw':1},whiskerprops={'color':COLORS[k]},capprops={'color':COLORS[k]})
    ax.axhline(0,color='#667085',ls='--',lw=.7)
    ax.set_xticks(range(3),['80%','60%','40%']);ax.set_xlim(-.55,2.55)
    ax.set_ylim(-85,45);ax.set_yticks([-80,-40,0,40]);ax.grid(axis='y',alpha=.15)
    ax.set_xlabel('Relative sampling intensity',fontsize=7,labelpad=3)
    ax.set_ylabel('Paired ratio change (%)',fontsize=7,labelpad=2)
    ax.tick_params(labelsize=7)
    fig.legend([Line2D([0],[0],color=COLORS[k],lw=1.5) for k in KINDS],
               ['Design-ignoring','Design-respecting'],loc='upper center',
               frameon=False,bbox_to_anchor=(.60,1.01),fontsize=7,labelspacing=.25)
    fig.subplots_adjust(left=.25,right=.98,bottom=.19,top=.77)
    save(fig,'fig1_shared_sampling')

def figures(draws,y,n,z,ids,months,factual):
    sampling_figures(draws)
    fig,ax=plt.subplots(2,1,figsize=(6.6,3.55),sharex=True)
    for a,axis in enumerate(ax):
        axis.plot(range(27),y[z==a].sum(0),color='#222',label='Observed',lw=1)
        for k in KINDS:axis.plot(range(27),factual[k]['fitted'][z==a].sum(0),color=COLORS[k],label='Design-'+k,lw=1)
        axis.set_ylabel('Positive events');axis.set_title(['Control','Intervention'][a],loc='left');axis.grid(axis='y',alpha=.15)
    ax[0].legend(frameon=False,ncol=3,fontsize=7)
    ax[-1].set_xticks([0,6,12,18,24],[months[i] for i in [0,6,12,18,24]]);ax[-1].set_xlabel('Illness-onset month')
    fig.tight_layout();save(fig,'factual_monthly_fit')
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.5),gridspec_kw={'width_ratios':[1.3,1]})
    for k,offset in zip(KINDS,[-.12,.12]):
        ax[0].plot(range(27),factual[k]['par'][2:-1],color=COLORS[k],label='Design-'+k,lw=1)
        ax[1].plot(np.arange(24)+offset,factual[k]['u'],'o',color=COLORS[k],ms=2.5)
    ax[0].set_xticks([0,12,24],['2018','2019','2020']);ax[0].set_xlabel('Illness-onset year');ax[0].set_ylabel('Fitted monthly state');ax[0].legend(frameon=False,fontsize=7)
    ax[1].axhline(0,color='gray',lw=.6);ax[1].axvline(11.5,color='gray',ls=':',lw=.6)
    ax[1].set_xticks([5.5,17.5],['Control clusters','Intervention clusters']);ax[1].set_ylabel('Cluster intercept mode')
    fig.tight_layout();save(fig,'latent_components')
    fig,ax=plt.subplots(6,4,figsize=(6.6,7.7),sharex=True)
    for c,axis in enumerate(ax.flat):
        axis.plot(range(27),y[c],color='#222',lw=.6)
        for k in KINDS:axis.plot(range(27),factual[k]['fitted'][c],color=COLORS[k],lw=.7)
        axis.set_title(f'Cluster {ids[c]} ({"control" if z[c]==0 else "intervention"})',fontsize=6.5,loc='left')
        axis.set_xticks([0,12,24],['18','19','20']);axis.tick_params(labelsize=6);axis.grid(axis='y',alpha=.1)
    fig.supylabel('Observed and fitted positive events',fontsize=8)
    fig.supxlabel('Illness-onset year (2018--2020)',fontsize=8)
    fig.legend([Line2D([0],[0],color=c,lw=1) for c in ['#222',COLORS['ignoring'],COLORS['respecting']]],['Observed','Design-ignoring','Design-respecting'],loc='upper center',ncol=3,frameon=False,fontsize=7)
    fig.tight_layout(rect=[.015,.015,1,.97],h_pad=1,w_pad=.8);save(fig,'factual_cluster_month_fit')

def main():
    started=time.perf_counter();y,n,z,ids,months=data();factual={k:fit(k,y,n,z) for k in KINDS}
    refs=[];rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for b,(rr,dd) in enumerate(pool.map(run_replicate,shared_tasks(y,n,z,factual))):
            refs.extend(rr);rows.extend(dd)
            if (b+1)%100==0:print(f'{b+1}/500',flush=True)
    elapsed=time.perf_counter()-started
    draws=pd.DataFrame(rows).rename(columns={'retention':'relative_sampling_intensity'})
    references=pd.DataFrame(refs)
    assert len(draws)==6000 and set(draws.scenario)==set(SCENARIOS)
    previous=pd.read_csv(ROOT/'outputs/cluster_month_analysis/draws.csv').rename(columns={'retention':'relative_sampling_intensity'})
    paired=draws.merge(previous,on=['replicate','scenario','relative_sampling_intensity','model'],suffixes=('_new','_old'),validate='one_to_one')
    assert np.allclose(paired.ratio_new,paired.ratio_old,rtol=1e-10)
    draws.to_csv(OUT/'draws.csv',index=False);references.to_csv(OUT/'bootstrap_references.csv',index=False)
    summary=draws.groupby(['scenario','relative_sampling_intensity','model']).relative_change_percent.agg(median='median',q025=lambda s:s.quantile(.025),q975=lambda s:s.quantile(.975)).reset_index()
    summary.to_csv(OUT/'sampling_summary.csv',index=False)
    sampling_table(summary)
    rows=[]
    for k,f in factual.items():
        v=references.loc[references.model==k,'ratio']
        rows.append({'model':k,'ratio':f['ratio'],'q025':v.quantile(.025),'q975':v.quantile(.975),'sigma_cluster':f['sigma_cluster'],
                     'cluster_month_rmse':np.sqrt(np.mean((y-f['fitted'])**2)),
                     'arm_month_display_rmse':np.sqrt(np.mean([(y[z==a].sum(0)-f['fitted'][z==a].sum(0))**2 for a in (0,1)]))})
    pd.DataFrame(rows).to_csv(OUT/'factual_summary.csv',index=False)
    np.savez(OUT/'factual_parameters.npz',**{k:f['par'] for k,f in factual.items()})
    # These are numerical checks of exactly the same factual / shared-scenario fitter.
    src=ROOT/'outputs/cluster_month_analysis'
    (OUT/'validation.json').write_bytes((src/'validation.json').read_bytes())
    sparse=json.loads((src/'sparse_quadrature_validation.json').read_text())
    sparse=[r for r in sparse if r['scenario'] in ('reference',*SCENARIOS)]
    for r in sparse:r['relative_sampling_intensity']=r.pop('retention')
    (OUT/'sparse_quadrature_validation.json').write_text(json.dumps(sparse,indent=2))
    figures(draws,y,n,z,ids,months,factual)
    metadata={'replicates':500,'scenarios':SCENARIOS,'relative_sampling_intensities':R,'seed':[20260910,0],
              'main_fit_count':7002,'elapsed_fitting_seconds':elapsed,'workers':4,'python':platform.python_version(),
              'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
              'max_gradient':float(draws.gradient_max.max()),'variance_lower_boundary':int(draws.variance_boundary.sum()),
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('cluster_month_analysis.py'),Path(__file__).with_name('sampling_policy.py')]}}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)
if __name__=='__main__':main()
