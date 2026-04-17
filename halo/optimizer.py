"""
HALO -- Homeostatic Adaptive Learning Optimizer.

Adam with a learnable brain that reads the loss landscape and reshapes
the update rule every step. The brain is trained by gradient descent using
information the optimizer already computes.
"""

import torch
from torch.optim import Optimizer

from .diagnostics import DiagnosticsTracker
from .meta_grads import compute_meta_grads


# Degree-1 biases for the three policy sigmoids (slope=0, so flat at init).
_PHI_BIAS = (1.4, 3.0, -2.0)

META_GRAD_CLIP = 1.0  # safety cap on ||meta_grad_acc|| before phi update


def _build_phi_init(degree: int) -> torch.Tensor:
    """Build the initial phi vector for a given polynomial degree.

    Layout: three consecutive blocks of (degree+1) coefficients, one per
    policy variable (pm, pv, ps). Within each block the coefficients are
    ordered highest-power-first: [coeff_d, ..., coeff_1, coeff_0].

    Higher-order coefficients start at zero; the degree-1 (slope) coefficient
    is zero; the degree-0 (bias) coefficient gets the Adam-like default.
    This makes the initial behavior identical regardless of degree.
    """
    n = degree + 1
    phi = torch.zeros(3 * n)
    for i, bias in enumerate(_PHI_BIAS):
        phi[i * n + n - 1] = bias  # constant term (last in block)
    return phi


def _horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a polynomial using Horner's method.

    coeffs: 1-D tensor [c_d, c_{d-1}, ..., c_1, c_0] (highest power first).
    x: scalar tensor.
    Returns: scalar tensor = c_d*x^d + ... + c_1*x + c_0.
    """
    out = coeffs[0]
    for k in range(1, coeffs.shape[0]):
        out = out * x + coeffs[k]
    return out


class HALO(Optimizer):
    r"""HALO optimizer.

    Args:
        params: iterable of parameters to optimize.
        lr: learning rate (default: 1e-3).
        betas: coefficients for running averages of gradient and its square (default: (0.9, 0.999)).
        eps: term for numerical stability (default: 1e-8).
        weight_decay: weight decay coefficient (default: 0.01).
        gamma: smoothing factor for the landscape signal (default: 0.99).
        eta_phi: meta learning rate for the response policy (default: 0.01).
        degree: polynomial degree for the response policy (default: 1).
        diagnostics: if True, record internal state every step (default: False).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        gamma=0.99,
        eta_phi=0.01,
        degree=1,
        diagnostics=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if not (isinstance(degree, int) and degree >= 1):
            raise ValueError(f"Invalid degree: {degree} (must be int >= 1)")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            eta_phi=eta_phi,
            degree=degree,
        )
        super().__init__(params, defaults)

        self.diagnostics = (
            DiagnosticsTracker(degree=degree) if diagnostics else None
        )

        phi_init = _build_phi_init(degree)

        # Initialize phi and rho per group. rho is a 0-dim tensor on the
        # first param's device to avoid GPU->CPU syncs in the hot path.
        for group in self.param_groups:
            device = next(iter(group["params"])).device
            group["rho"] = torch.zeros((), device=device)
            group["phi"] = phi_init.clone().to(device=device)
            group["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]
            eta_phi = group["eta_phi"]

            phi = group["phi"]
            rho = group["rho"]
            group["step"] += 1
            step = group["step"]

            bias_corr1 = 1 - beta1 ** step
            bias_corr2 = 1 - beta2 ** step

            # ---- Pass 1: single landscape signal from the full gradient ----
            # Accumulate dot(g, g_prev), ||g||^2, ||g_prev||^2 across all params.
            # Also handles lazy state init so pass 2 can rely on it.
            dot = torch.zeros((), device=phi.device)
            sq_g = torch.zeros((), device=phi.device)
            sq_prev = torch.zeros((), device=phi.device)

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["g_prev"] = torch.zeros_like(p)
                    state["intermediates"] = None
                g = p.grad
                g_prev = state["g_prev"]
                dot = dot + (g * g_prev).sum().to(phi.device)
                sq_g = sq_g + g.pow(2).sum().to(phi.device)
                sq_prev = sq_prev + g_prev.pow(2).sum().to(phi.device)

            denom_cos = (sq_g * sq_prev).sqrt()
            r = torch.where(
                denom_cos > 1e-12,
                dot / denom_cos.clamp(min=1e-12),
                torch.zeros((), device=phi.device),
            )
            rho = gamma * rho + (1 - gamma) * r

            # ---- Single response policy for the whole group ----
            degree = group["degree"]
            n_phi = degree + 1
            coeffs_m = phi[:n_phi]
            coeffs_v = phi[n_phi : 2 * n_phi]
            coeffs_s = phi[2 * n_phi :]

            z_m = _horner(coeffs_m, rho)
            z_v = _horner(coeffs_v, rho)
            z_s = _horner(coeffs_s, rho)

            pm = torch.sigmoid(z_m)
            pv = 0.5 * torch.sigmoid(z_v)
            ps = torch.sigmoid(z_s)

            # ---- Pass 2: meta-gradient + Adam update for each param ----
            meta_grad_acc = torch.zeros(3 * n_phi, device=phi.device)

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                m = state["m"]
                v = state["v"]
                g_prev = state["g_prev"]  # still holds g_{t-1} until we copy below

                # Meta-gradient from step t (intermediates) against g_{t+1} (current g).
                intermediates = state.get("intermediates")
                if intermediates is not None:
                    mg = compute_meta_grads(
                        g, intermediates, g_prev, phi, lr, degree=degree
                    )
                    meta_grad_acc += mg.to(phi.device)

                # Adam bookkeeping
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / bias_corr1
                v_hat = v / bias_corr2

                # Move policy scalars to param device if needed (no-op on single device).
                pm_p = pm.to(g.device) if pm.device != g.device else pm
                pv_p = pv.to(g.device) if pv.device != g.device else pv
                ps_p = ps.to(g.device) if ps.device != g.device else ps

                # Adaptive step
                n = pm_p * m_hat + (1 - pm_p) * g
                n_abs = torch.clamp(n.abs(), min=1e-12)
                denom = v_hat.pow(pv_p) + eps
                d = -lr * n_abs.pow(1 - ps_p) * torch.sign(n) / denom

                # Decoupled weight decay (AdamW-style)
                if weight_decay != 0:
                    d = d - lr * weight_decay * p.data

                p.data.add_(d)

                # Store intermediates for next step's meta-gradient.
                rho_p = rho.to(g.device) if rho.device != g.device else rho
                state["intermediates"] = {
                    "n_t": n.detach(),
                    "m_hat": m_hat.detach(),
                    "v_hat": v_hat.detach(),
                    "p_m": pm_p.detach(),
                    "p_v": pv_p.detach(),
                    "p_s": ps_p.detach(),
                    "rho": rho_p.detach().clone(),
                }

                # Update g_prev to g_t for the next step.
                g_prev.copy_(g)

            # ---- Update phi with accumulated meta-gradients ----
            # Clip by L2 norm to guard against log|n| blow-up in meta-gradients.
            mg_norm = meta_grad_acc.norm()
            scale = torch.clamp(META_GRAD_CLIP / (mg_norm + 1e-12), max=1.0)
            phi.add_(meta_grad_acc * scale, alpha=-eta_phi)

            # Persist updated rho (still a tensor, no sync).
            group["rho"] = rho

            # ---- Diagnostics ----
            if self.diagnostics is not None:
                self.diagnostics.record(
                    step,
                    float(r),
                    float(rho),
                    float(pm),
                    float(pv),
                    float(ps),
                    phi,
                )

        return loss
