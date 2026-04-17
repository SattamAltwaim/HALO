"""
Analytic meta-gradient computation for the HALO optimizer.

Pure function that computes the 3*(degree+1) meta-gradients (dL/dphi_i)
from stored detached intermediates. No autograd graph, no state.
"""

import torch


def _safe_pow(base, exp_val):
    """|base|^exp_val with clamping for numerical stability."""
    return torch.clamp(base.abs(), min=1e-12).pow(exp_val)


def _safe_log(x):
    """log(|x|) with clamping for numerical stability."""
    return torch.log(torch.clamp(x.abs(), min=1e-12))


def _horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a polynomial via Horner's method (highest power first)."""
    out = coeffs[0]
    for k in range(1, coeffs.shape[0]):
        out = out * x + coeffs[k]
    return out


def compute_meta_grads(g_next, intermediates, g_t, phi, eta, *, degree=1):
    """Compute the 3*(degree+1) meta-gradients dL/dphi_i for one param tensor.

    The chain: phi -> (pm, pv, ps) -> d_t -> theta_{t+1} -> g_{t+1} -> L

        dL/dphi_i = g_{t+1}^T * dd_t/dphi_i  (summed over all elements)

    Args:
        g_next: gradient at step t+1, shape matches param.
        intermediates: dict with keys n_t, m_hat, v_hat, p_m, p_v, p_s, rho.
        g_t: gradient at step t (stored separately as state["g_prev"]).
        phi: current group-level phi vector, length 3*(degree+1).
        eta: main optimizer learning rate.
        degree: polynomial degree of the response policy.

    Returns:
        grads: tensor of shape (3*(degree+1),).
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

    # Scalar dL/dp for each policy variable
    dL_dpm = torch.sum(g_next * dd_dpm)
    dL_dpv = torch.sum(g_next * dd_dpv)
    dL_dps = torch.sum(g_next * dd_dps)

    # --- Chain rule through sigmoid and polynomial ---
    n_phi = degree + 1
    coeffs_m = phi[:n_phi]
    coeffs_v = phi[n_phi : 2 * n_phi]
    coeffs_s = phi[2 * n_phi :]

    z_m = _horner(coeffs_m, rho)
    z_v = _horner(coeffs_v, rho)
    z_s = _horner(coeffs_s, rho)

    s_m = torch.sigmoid(z_m)
    s_v = torch.sigmoid(z_v)
    s_s = torch.sigmoid(z_s)

    ds_m = s_m * (1 - s_m)
    ds_v = s_v * (1 - s_v)
    ds_s = s_s * (1 - s_s)

    # Build rho powers: [rho^d, rho^{d-1}, ..., rho^1, rho^0]
    rho_powers = torch.empty(n_phi, device=phi.device)
    rho_powers[n_phi - 1] = 1.0
    for k in range(n_phi - 2, -1, -1):
        rho_powers[k] = rho_powers[k + 1] * rho

    # dL/d(coeffs_m[k]) = dL/dpm * ds_m * rho^(d-k)
    grads_m = dL_dpm * ds_m * rho_powers
    # pv = 0.5 * sigmoid(z_v), so dp_v/dz_v = 0.5 * sigmoid'(z_v)
    grads_v = dL_dpv * 0.5 * ds_v * rho_powers
    grads_s = dL_dps * ds_s * rho_powers

    return torch.cat([grads_m, grads_v, grads_s])
