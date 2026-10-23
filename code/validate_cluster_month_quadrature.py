"""Check Laplace fits against adaptive quadrature on a preselected sparse case.
Uses bootstrap replicate 0 and 40% retention in all three scenarios, without
selection by fitted effects. Run after cluster_month_analysis.py.
"""
import json
import numpy as np
from scipy.optimize import minimize
from cluster_month_analysis import OUT, KINDS, data, fit, tasks, aghq_objective


def main():
    y,n,z,_,_=data()
    factual={k:fit(k,y,n,z) for k in KINDS}
    task=next(tasks(y,n,z,factual,1))
    _,yb,nb,z,_,_,settings,_=task
    scenarios=[('reference',yb,nb)]+next(s for r,s in settings if r==.4)
    rows=[]
    for scenario,ys,ns in scenarios:
        for k in KINDS:
            f=fit(k,ys,ns,z,factual[k]['par'])
            ag=minimize(aghq_objective,f['par'],args=(k,ys,ns,z,25),method='L-BFGS-B',
                bounds=[(None,None)]*(len(f['par'])-1)+[(-7,2)],options={'ftol':1e-11,'gtol':1e-5,'maxiter':1000})
            assert ag.success,ag.message
            quadrature_error=abs(aghq_objective(ag.x,k,ys,ns,z,25)-aghq_objective(ag.x,k,ys,ns,z,45))
            assert quadrature_error<1e-5,quadrature_error
            rows.append({'replicate':0,'retention':1. if scenario=='reference' else .4,'scenario':scenario,'model':k,
                         'laplace_ratio':float(f['ratio']),'aghq25_ratio':float(np.exp(ag.x[1])),
                         'relative_ratio_difference_percent':float(100*np.expm1(ag.x[1]-f['beta'])),
                         'quadrature25_vs45_objective_difference':quadrature_error})
    for row in rows:
        ref=next(r for r in rows if r['scenario']=='reference' and r['model']==row['model'])
        row['laplace_paired_change_percent']=100*(row['laplace_ratio']/ref['laplace_ratio']-1)
        row['aghq25_paired_change_percent']=100*(row['aghq25_ratio']/ref['aghq25_ratio']-1)
    (OUT/'sparse_quadrature_validation.json').write_text(json.dumps(rows,indent=2))
    print('Maximum ratio discrepancy (%):',max(abs(r['relative_ratio_difference_percent']) for r in rows))
    print('Maximum paired-change discrepancy (percentage points):',max(abs(r['laplace_paired_change_percent']-r['aghq25_paired_change_percent']) for r in rows))

if __name__=='__main__': main()
