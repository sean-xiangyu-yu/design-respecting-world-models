"""Sampling-policy helpers for the relative sampling-intensity experiment (Appendix A.1).

policy_path draws the logit-scale AR(1) monthly path g_t; calibrated_q sets the
intercept so that the enrollment-weighted expected sampled fraction equals r.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

POLICY_RHO = 0.7
POLICY_SD = 0.8  # Stationary SD on the logit scale; imposed, not estimated.


def policy_path(rng, size):
    path = np.empty(size)
    path[0] = rng.normal(0, POLICY_SD)
    for t in range(1, size):
        path[t] = POLICY_RHO * path[t - 1] + rng.normal(
            0, POLICY_SD * np.sqrt(1 - POLICY_RHO**2)
        )
    return path


def calibrated_q(logit_offsets, counts, retention):
    # Expected retained proportion, weighted by all original enrolled events.
    weights = counts / counts.sum()
    intercept = brentq(
        lambda z: np.sum(weights * expit(z + logit_offsets)) - retention,
        -30, 30,
    )
    q = np.broadcast_to(expit(intercept + logit_offsets), counts.shape).copy()
    assert np.isclose(np.sum(weights * q), retention, atol=1e-10)
    assert np.all((q > 0) & (q < 1))
    return q
