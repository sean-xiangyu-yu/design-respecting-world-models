"""Cluster-month factual fits, numerical validation, and paired sampling experiment.

Conditional Poisson / binomial emissions, shared AR(1) monthly state, independent
N(0, sigma_cluster**2) cluster intercepts. Cluster intercepts are integrated by
Laplace approximation; sigma_cluster is estimated anew in every fit. Monthly
states retain the original MAP AR(1) penalty. Whole-cluster bootstrap copies are
separate pseudo-clusters, each keeping its complete trajectory.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cluster_month_analysis'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import minimize, approx_fprime
from scipy.special import expit, logsumexp
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sampling_policy import policy_path, calibrated_q

RHO, STATE_PRECISION, RIDGE = .9, 5., 1e-4
KINDS = ('ignoring', 'respecting')
RETENTION = (.8, .6, .4)
SCENARIOS = ('shared_monthly', 'positive_specific', 'shared_cluster_monthly')
COLORS = {'ignoring': '#D56516', 'respecting': '#168B75'}


def data():
    events = pd.read_csv(ROOT / 'data' / 'df_all_participant.csv').drop_duplicates('participant_id')
    events['month'] = pd.to_datetime(events.illness_onset).dt.to_period('M').astype(str)
    months = sorted(events.month.unique())
    arms = events.groupby('cluster').intervention.first().sort_index()
    ids = np.concatenate([arms.index[arms == a].to_numpy() for a in (0, 1)])
    idx = pd.MultiIndex.from_product([ids, months], names=['cluster', 'month'])
    cm = events.groupby(['cluster', 'month']).dengue.agg(['sum', 'size']).reindex(idx, fill_value=0)
    y, n = (cm[col].to_numpy(float).reshape(24, 27) for col in ('sum', 'size'))
    assert len(events) == 6306 and y.sum() == 385 and n.sum() == 6306
    return y, n, np.repeat([0., 1.], 12), ids, months


def terms(kind, eta, y, n):
    if kind == 'ignoring':
        mu = np.exp(eta)
        return mu-y*eta, mu-y, mu, mu
    p = expit(eta)
    w = n*p*(1-p)
    return n*np.logaddexp(0, eta)-y*eta, n*p-y, w, w*(1-2*p)


def cluster_modes(kind, base, y, n, precision):
    u = np.zeros(len(y))
    for _ in range(100):
        loss, r, w, v = terms(kind, base+u[:, None], y, n)
        h = w.sum(axis=1)+precision
        step = (r.sum(axis=1)+precision*u)/h
        u -= np.clip(step, -2., 2.)
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        raise RuntimeError('Cluster mode Newton iteration did not converge')
    loss, r, w, v = terms(kind, base+u[:, None], y, n)
    return u, loss, r, w, v, w.sum(axis=1)+precision


def penalty(par):
    x = par[2:-1]
    innovations = np.r_[x[0], x[1:]-RHO*x[:-1]]
    value = .5*STATE_PRECISION*(innovations@innovations) + .5*RIDGE*(par[:2]@par[:2])
    grad = STATE_PRECISION*innovations
    grad[:-1] -= STATE_PRECISION*RHO*innovations[1:]
    return value, grad


def objective(par, kind, y, n, z):
    base = par[0]+par[1]*z[:, None]+par[2:-1][None, :]
    precision = np.exp(-2*par[-1])
    u, loss, r, w, v, h = cluster_modes(kind, base, y, n, precision)
    value = loss.sum()+.5*precision*(u@u)+len(y)*par[-1]+.5*np.log(h).sum()
    # Exact gradient of the Laplace objective, including mode dependence.
    vsum = v.sum(axis=1)
    geta = r + .5*(v-w*(vsum/h)[:, None])/h[:, None]
    gsigma = len(y)-precision*(u@u)+.5*np.sum((vsum*2*precision*u/h-2*precision)/h)
    pen, gx = penalty(par)
    grad = np.r_[geta.sum()+RIDGE*par[0], (geta*z[:, None]).sum()+RIDGE*par[1], geta.sum(axis=0)+gx, gsigma]
    return float(value+pen), grad


def fit(kind, y, n, z, warm=None):
    if warm is None:
        par = np.zeros(y.shape[1]+3)
        par[0] = np.log((y.sum()+.1)/(y.size if kind=='ignoring' else n.sum()-y.sum()+.1))
        par[-1] = np.log(.5)
    else:
        par = np.array(warm, copy=True)
    bounds = [(None, None)]*(len(par)-1)+[(-7., 2.)]
    result = minimize(objective, par, args=(kind,y,n,z), jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'ftol':1e-11, 'gtol':2e-6, 'maxiter':1500, 'maxls':40})
    if not result.success or np.max(np.abs(result.jac[:-1])) > .003:
        result = minimize(objective, result.x, args=(kind,y,n,z), jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'ftol':1e-13, 'gtol':1e-6, 'maxiter':2500, 'maxls':60})
    assert result.success, result.message
    assert np.max(np.abs(result.jac[:-1])) < .01, result.jac
    par = result.x
    base = par[0]+par[1]*z[:,None]+par[2:-1][None,:]
    u = cluster_modes(kind,base,y,n,np.exp(-2*par[-1]))[0]
    eta = base+u[:,None]
    fitted = np.exp(eta) if kind=='ignoring' else n*expit(eta)
    return {'par':par, 'beta':par[1], 'ratio':np.exp(par[1]), 'sigma_cluster':np.exp(par[-1]),
            'u':u, 'fitted':fitted, 'objective':result.fun, 'iterations':result.nit,
            'gradient_max':float(np.max(np.abs(result.jac[:-1]))), 'variance_boundary':bool(par[-1]<-6.99)}


def aghq_objective(par, kind, y, n, z, nodes=25):
    base = par[0]+par[1]*z[:,None]+par[2:-1][None,:]
    precision = np.exp(-2*par[-1])
    u, _, _, _, _, h = cluster_modes(kind,base,y,n,precision)
    q, weight = hermgauss(nodes)
    at = u[:,None]+np.sqrt(2/h[:,None])*q[None,:]
    eta = base[:,None,:]+at[:,:,None]
    loss = terms(kind,eta,y[:,None,:],n[:,None,:])[0].sum(axis=2)
    exponent = -loss-.5*precision*at**2+q[None,:]**2+np.log(weight)[None,:]
    # Normalized normal prior and adaptive quadrature Jacobian.
    log_integral = logsumexp(exponent,axis=1)-par[-1]-.5*np.log(np.pi*h)
    return float(-log_integral.sum()+penalty(par)[0])


def run_replicate(task):
    b, y, n, z, path, cluster_effect, uniforms_seed, warms = task
    # Data and policy draws are made serially below to match the old experiment.
    baseline = {k:fit(k,y,n,z,warms[k]) for k in KINDS}
    refs = [{'replicate':b,'model':k,'ratio':f['ratio'],'sigma_cluster':f['sigma_cluster'],
             'gradient_max':f['gradient_max'],'variance_boundary':f['variance_boundary']} for k,f in baseline.items()]
    rows=[]
    for retention, scenarios in uniforms_seed:
        case_cache={}
        for scenario, yp, nn in scenarios:
            for kind in KINDS:
                if kind=='ignoring' and scenario=='positive_specific':
                    f=case_cache['shared_monthly']
                else:
                    f=fit(kind,yp,nn,z,baseline[kind]['par'])
                if kind=='ignoring': case_cache[scenario]=f
                delta=f['beta']-baseline[kind]['beta']
                rows.append({'replicate':b,'scenario':scenario,'retention':retention,'model':kind,
                             'ratio':f['ratio'],'paired_reference_ratio':baseline[kind]['ratio'],
                             'log_ratio_change':delta,'relative_change_percent':100*np.expm1(delta),
                             'sigma_cluster':f['sigma_cluster'],'gradient_max':f['gradient_max'],
                             'variance_boundary':f['variance_boundary'],
                             'retained_positive':int(yp[12:].sum()),'retained_negative':int((nn-yp)[12:].sum())})
    return refs,rows


def tasks(y,n,z,factual,B):
    rng=np.random.default_rng(np.random.SeedSequence([20260910,0]))
    pos,neg=[y[:12],y[12:]],[n[:12]-y[:12],n[12:]-y[12:]]
    for b in range(B):
        selected=[rng.choice(12,12,replace=True) for _ in (0,1)]
        pb=[pos[a][selected[a]] for a in (0,1)]
        nb=[neg[a][selected[a]] for a in (0,1)]
        yy=np.concatenate(pb); nn=yy+np.concatenate(nb)
        path=policy_path(rng,27)
        ce=rng.normal(0,.8,12)[selected[1],None]
        settings=[]
        for retention in RETENTION:
            q=calibrated_q(path,pb[1]+nb[1],retention)
            yp=rng.binomial(pb[1].astype(int),q); yn=rng.binomial(nb[1].astype(int),q)
            qc=calibrated_q(path+ce,pb[1]+nb[1],retention)
            ypc=rng.binomial(pb[1].astype(int),qc); ync=rng.binomial(nb[1].astype(int),qc)
            scenarios=[]
            for scenario,ps,ns in [('shared_monthly',yp,yn),('positive_specific',yp,nb[1]),('shared_cluster_monthly',ypc,ync)]:
                ys=np.concatenate([pb[0],ps]); nsall=ys+np.concatenate([nb[0],ns])
                assert np.all(ys<=yy) and np.all(nsall<=nn)
                scenarios.append((scenario,ys,nsall))
            settings.append((retention,scenarios))
        yield b,yy,nn,z,path,ce,settings,{k:factual[k]['par'] for k in KINDS}


def plots(draws,y,n,z,months,factual):
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,scenario,title in zip(axes,SCENARIOS,['Shared monthly sampling','Positive-only sampling','Shared cluster-month sampling']):
        for j,r in enumerate(RETENTION):
            for kind,offset in zip(KINDS,[-.17,.17]):
                values=draws.loc[(draws.scenario==scenario)&(draws.retention==r)&(draws.model==kind),'relative_change_percent']
                ax.boxplot([values],positions=[j+offset],widths=.26,whis=(2.5,97.5),showfliers=False,patch_artist=True,
                           boxprops={'facecolor':COLORS[kind],'alpha':.3},medianprops={'color':COLORS[kind]})
        ax.axhline(0,color='gray',ls='--'); ax.set_xticks(range(3),['80%','60%','40%']); ax.set_title(title)
        ax.set_xlabel('Expected retention setting'); ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Paired change in fitted conditional ratio (%)')
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0],[0],color=COLORS[k],lw=3) for k in KINDS],['Design-ignoring','Design-respecting'],loc='upper center',ncol=2)
    fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(OUT/'sampling_comparison.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for a,ax in enumerate(axes):
        ax.plot(months,y[z==a].sum(axis=0),color='black',label='Observed',lw=1.5)
        for k in KINDS: ax.plot(months,factual[k]['fitted'][z==a].sum(axis=0),color=COLORS[k],label=k)
        ax.set_title(['Control','Intervention'][a]); ax.set_ylabel('Positive events'); ax.legend()
    axes[1].set_xticks(range(0,27,4),months[::4]); fig.tight_layout(); fig.savefig(OUT/'factual_arm_month_display.png',dpi=170);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=500);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    started=time.perf_counter(); y,n,z,ids,months=data()
    factual={k:fit(k,y,n,z) for k in KINDS}
    validation={}
    for k,f in factual.items():
        # Validate analytic derivatives away from the optimum, across all coordinates.
        test=f['par']+np.random.default_rng(7).normal(0,.03,len(f['par']))
        g=objective(test,k,y,n,z)[1]
        gn=np.empty_like(g)
        for j in range(len(g)):
            d=np.zeros_like(g);d[j]=1e-5
            gn[j]=(objective(test+d,k,y,n,z)[0]-objective(test-d,k,y,n,z)[0])/2e-5
        err=float(np.max(np.abs(g-gn))); assert err<1e-5,(k,err)
        alternative=fit(k,y,n,z,np.r_[f['par'][:-1],np.log(1.2)])
        assert abs(alternative['beta']-f['beta'])<.001
        # Higher-accuracy adaptive quadrature refit checks the Laplace approximation.
        ag=minimize(aghq_objective,f['par'],args=(k,y,n,z,25),method='L-BFGS-B',bounds=[(None,None)]*(len(f['par'])-1)+[(-7,2)],
                    options={'ftol':1e-11,'gtol':1e-5,'maxiter':1000})
        assert ag.success,ag.message
        validation[k]={'max_gradient_error':err,'laplace_ratio':float(f['ratio']), 'aghq25_ratio':float(np.exp(ag.x[1])),
                       'aghq25_sigma_cluster':float(np.exp(ag.x[-1])), 'quadrature25_vs45_objective_difference':abs(aghq_objective(ag.x,k,y,n,z,25)-aghq_objective(ag.x,k,y,n,z,45))}
        print(k,validation[k],flush=True)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    facts=[];predictions=[]
    for k,f in factual.items():
        facts.append({'model':k,'ratio':f['ratio'],'sigma_cluster':f['sigma_cluster'],
                      'cluster_month_rmse':float(np.sqrt(np.mean((y-f['fitted'])**2))),
                      'arm_month_rmse':float(np.sqrt(np.mean([(y[z==a].sum(axis=0)-f['fitted'][z==a].sum(axis=0))**2 for a in (0,1)]))),
                      'gradient_max':f['gradient_max']})
        for c in range(24):
            for t in range(27):predictions.append({'model':k,'cluster':ids[c],'month':months[t],'arm':int(z[c]),'positive':y[c,t],'total':n[c,t],'fitted_positive':f['fitted'][c,t],'cluster_intercept_mode':f['u'][c],'time_state':f['par'][2+t]})
    pd.DataFrame(facts).to_csv(OUT/'factual_results.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'factual_predictions.csv',index=False)
    np.savez(OUT/'factual_parameters.npz',**{k:f['par'] for k,f in factual.items()})
    if args.validate_only:return
    refs=[];rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for b,(ref,row) in enumerate(pool.map(run_replicate,tasks(y,n,z,factual,args.replicates))):
            refs.extend(ref);rows.extend(row)
            if (b+1)%25==0:print(f'{b+1}/{args.replicates}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    draws=pd.DataFrame(rows); reference=pd.DataFrame(refs)
    draws.to_csv(OUT/'draws.csv',index=False);reference.to_csv(OUT/'bootstrap_references.csv',index=False)
    summary=draws.groupby(['scenario','retention','model']).relative_change_percent.agg(median='median',q025=lambda v:v.quantile(.025),q975=lambda v:v.quantile(.975)).reset_index()
    summary.to_csv(OUT/'sampling_summary.csv',index=False)
    ci=reference.groupby('model').ratio.agg(q025=lambda v:v.quantile(.025),q975=lambda v:v.quantile(.975)).reset_index()
    ci.to_csv(OUT/'factual_bootstrap_intervals.csv',index=False)
    assert len(draws)==args.replicates*18
    plots(draws,y,n,z,months,factual)
    metadata={'replicates':args.replicates,'seed':[20260910,0],'fitting':'Laplace marginal likelihood over cluster intercepts; MAP monthly states',
              'state_rho':RHO,'state_precision':STATE_PRECISION,'fixed_effect_ridge':RIDGE,'cluster_sd_estimated_each_fit':True,
              'cluster_sd_log_bounds':[-7,2],'bootstrap_duplicate_clusters':'new pseudo-cluster identities, whole trajectories retained',
              'conditional_ratio_interpretation':{'ignoring':'conditional positive-count ratio','respecting':'conditional positive-label odds ratio'},
              'max_gradient':float(draws.gradient_max.max()),'variance_boundary_fits':int(draws.variance_boundary.sum()),
              'elapsed_seconds':time.perf_counter()-started}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(facts).to_string(index=False),flush=True);print(ci.to_string(index=False),flush=True)
    print(summary[summary.retention==.6].to_string(index=False),flush=True)

if __name__=='__main__':main()
