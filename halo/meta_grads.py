"""
Analytic meta-gradient computation for the HALO optimizer.

Pure function that computes the 6 meta-gradients (dL/dphi_i) from stored
detached intermediates. No autograd graph, no state.
"""

import torch


def _safe_pow(base, exp_val):
    """|base|^exp_val with clamping for numerical stability."""
    return torch.clamp(base.abs(), min=1e-12).pow(exp_val)


def _safe_log(x):
    """log(|x|) with clamping for numerical stability."""
    return torch.log(torch.clamp(x.abs(), min=1e-12))


def compute_meta_grads(g_next, intermediates, g_t, phi, eta):
    """Compute the 6 meta-gradients dL/dphi_i for a single parameter tensor.

    The chain: phi -> (pm, pv, ps) -> d_t -> theta_{t+1} -> g_{t+1} -> L

        dL/dphi_i = g_{t+1}^T * dd_t/dphi_i  (summed over all elements)

    Args:
        g_next: gradient at step t+1, shape matches param.
        intermediates: dict with keys n_t, m_hat, v_hat, p_m, p_v, p_s, rho.
        g_t: gradient at step t (stored separately as state["g_prev"]).
        phi: current group-level phi vector (6,).
        eta: main optimizer learning rate.

    Returns:
        grads: tensor of shape (6,) = [dL/da1, dL/db1, dL/da2, dL/db2, dL/da3, dL/db3]
    """
    n_t = intermediates["n_t"]
    m_hat = intermediates["m_hat"]
    v_hat = intermediates["v_hat"]
    pm = intermediates["p_m"]
    pv = intermediates["p_v"]
    ps = intermediates["p_s"]
    rho = intermediates["rho"]

    denom = _safe_pow(v_hat, pv) + 1e-8
    n_abs_neg_ps = _safe_pow(n_t, -ps)
    n_abs_1ps = _safe_pow(n_t, 1 - ps)
    sign_n = torch.sign(n_t)

    # dd_t / dp_m = -eta * (1 - ps) * |n|^(-ps) * (m_hat - g_t) / denom
    dd_dpm = -eta * (1 - ps) * n_abs_neg_ps * (m_hat - g_t) / denom

    # dd_t / dp_v = eta * |n|^(1-ps) * sign(n) * v^pv * log|v| / denom^2
    dd_dpv = eta * n_abs_1ps * sign_n * _safe_pow(v_hat, pv) * _safe_log(v_hat) / (denom * denom)

    # dd_t / dp_s = eta * |n|^(1-ps) * sign(n) * log|n| / denom
    dd_dps = eta * n_abs_1ps * sign_n * _safe_log(n_t) / denom

    # Dot products with g_next
    dL_dpm = torch.sum(g_next * dd_dpm)
    dL_dpv = torch.sum(g_next * dd_dpv)
    dL_dps = torch.sum(g_next * dd_dps)

    # Chain rule through sigmoid derivatives
    a1, b1, a2, b2, a3, b3 = phi.unbind()

    s1 = torch.sigmoid(a1 * rho + b1)
    s2 = torch.sigmoid(a2 * rho + b2)
    s3 = torch.sigmoid(a3 * rho + b3)

    ds1 = s1 * (1 - s1)
    ds2 = s2 * (1 - s2)
    ds3 = s3 * (1 - s3)

    # dp/dphi for each parameter
    dp_m_da1 = ds1 * rho
    dp_m_db1 = ds1
    dp_v_da2 = 0.5 * ds2 * rho
    dp_v_db2 = 0.5 * ds2
    dp_s_da3 = ds3 * rho
    dp_s_db3 = ds3

    # Final meta-gradients
    dL_da1 = dL_dpm * dp_m_da1
    dL_db1 = dL_dpm * dp_m_db1
    dL_da2 = dL_dpv * dp_v_da2
    dL_db2 = dL_dpv * dp_v_db2
    dL_da3 = dL_dps * dp_s_da3
    dL_db3 = dL_dps * dp_s_db3

    return torch.stack([dL_da1, dL_db1, dL_da2, dL_db2, dL_da3, dL_db3])
