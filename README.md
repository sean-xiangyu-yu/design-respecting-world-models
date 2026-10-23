# Design-Ignoring versus Design-Respecting World Models for Epidemiology

Code for the NeurIPS 2026 Workshop on World Models for High-Stakes Health paper
*Design-Ignoring versus Design-Respecting World Models for Epidemiology*
(Xiangyu Yu, Weiyu Liu).

The analysis fits two cluster-month state-space models to the AWED
cluster-randomized test-negative trial: a design-ignoring Poisson model of
recorded positive counts and a design-respecting binomial test-negative
observation model, sharing an AR(1) monthly state and normal cluster
intercepts (integrated by Laplace approximation). It then compares the paired
change in each model's fitted contrast when intervention-arm events are
sampled with a probability shared by test-positive and test-negative events.

## Contents

| File | Purpose |
|---|---|
| `code/cluster_month_analysis.py` | Data construction, model fitter, gradient and quadrature checks, 500-replicate paired sampling experiment (seed `[20260910, 0]`) |
| `code/validate_cluster_month_quadrature.py` | Laplace versus adaptive Gauss–Hermite check on a preselected sparse replicate (Appendix B.4) |
| `code/shared_sampling_manuscript.py` | Figures and tables reported in the paper |
| `code/sampling_policy.py` | Monthly sampling path and intensity calibration (Appendix A.1) |
| `code/synthetic_example.py` | The same check on synthetic data; runs in seconds without AWED records |
| `run_all.sh` | Runs the full analysis in order |

## Quick check without AWED data

```bash
pip install -r requirements.txt
cd code
python synthetic_example.py
```

Expected output: the design-ignoring contrast falls by roughly 20%, 40%, and 60%
at relative sampling intensities 0.8, 0.6, and 0.4, while the design-respecting
contrast stays near zero.

## Reproducing the paper

1. Place the AWED analysis file at `data/df_all_participant.csv` (see `data/README.md`).
2. Run `bash run_all.sh` (about 1.5 minutes with two CPU workers).

Outputs are written to `outputs/`:

| Paper item | File |
|---|---|
| Figure 1 | `outputs/cluster_month_shared_sampling/fig1_shared_sampling.pdf` |
| Table 1 (factual contrasts, bootstrap intervals, RMSE) | `outputs/cluster_month_shared_sampling/factual_summary.csv` |
| Figure 2 | `outputs/cluster_month_shared_sampling/factual_monthly_fit.pdf` |
| Table 2 | `outputs/cluster_month_shared_sampling/sampling_table.tex` |
| Figure 3 | `outputs/cluster_month_shared_sampling/cluster_month_sampling.pdf` |
| Appendix B.4 numerical checks | `outputs/cluster_month_analysis/validation.json`, `sparse_quadrature_validation.json` |
| Run metadata (software versions, fit counts, timing) | `outputs/cluster_month_shared_sampling/metadata.json` |

`cluster_month_analysis.py` also fits a positive-only sampling scenario that is
not reported in the paper; the paper's figures and tables use only the two
shared-sampling scenarios selected in `shared_sampling_manuscript.py`.

## License

MIT (see `LICENSE`). The license covers the code only; AWED participant-level
records are not included in this repository.
