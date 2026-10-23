"""Run the paired shared-sampling check on synthetic data (no AWED records needed).

Simulates AWED-sized data sets (24 clusters x 27 months, conditional odds
ratio 0.25) in which test-positive and test-negative counts share cluster-month
ascertainment, fits the design-ignoring (Poisson count) and design-respecting
(binomial TND) models with the same fitter as the main analysis, applies shared
monthly sampling to the intervention arm at r = 0.8, 0.6, 0.4, refits, and
prints median paired changes.  Usage: python synthetic_example.py [--datasets 20] [--seed 1]
"""
import argparse
import numpy as np
from cluster_month_analysis import fit
from sampling_policy import policy_path, calibrated_q


def ar1(rng, sd, rho, n):
    x = np.empty(n); x[0] = rng.normal(0, sd)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, sd * np.sqrt(1 - rho**2))
    return x


def simulate(rng, clusters=24, months=27, odds_ratio=.25):
    z = np.repeat([0., 1.], clusters // 2)
    x, v = ar1(rng, .9, .9, months), ar1(rng, .6, .9, months)        # disease and care-seeking time trends
    u, w = rng.normal(0, .4, clusters), rng.normal(0, .65, clusters)  # cluster heterogeneity
    lam_neg = np.exp(np.log(9.1) - .39 + v[None, :] + w[:, None])     # test-negative illness x ascertainment
    eta = np.log(.093 / .907) - .49 + np.log(odds_ratio) * z[:, None] + x[None, :] + u[:, None]
    y_neg = rng.poisson(lam_neg)
    y_pos = rng.poisson(lam_neg * np.exp(eta))
    return y_pos.astype(float), (y_pos + y_neg).astype(float), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=int, default=20); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    changes = {(k, r): [] for k in ('ignoring', 'respecting') for r in (.8, .6, .4)}
    for d in range(a.datasets):
        y, n, z = simulate(rng)
        ref = {k: fit(k, y, n, z) for k in ('ignoring', 'respecting')}
        path = policy_path(rng, y.shape[1])
        half = len(z) // 2
        for r in (.8, .6, .4):
            q = calibrated_q(path, n[half:], r)
            ys = np.r_[y[:half], rng.binomial(y[half:].astype(int), q)]
            ns = ys + np.r_[(n - y)[:half], rng.binomial((n - y)[half:].astype(int), q)]
            for k in ref:
                changes[k, r].append(100 * np.expm1(fit(k, ys, ns, z, ref[k]['par'])['beta'] - ref[k]['beta']))
    print(f'Median paired change over {a.datasets} synthetic AWED-sized data sets (true conditional OR 0.25):')
    for r in (.8, .6, .4):
        print(f'  r = {r:.1f}: design-ignoring {np.median(changes["ignoring", r]):+6.1f}%   '
              f'design-respecting {np.median(changes["respecting", r]):+6.1f}%')


if __name__ == '__main__':
    main()
