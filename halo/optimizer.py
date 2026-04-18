"""
HALO -- Homeostatic Adaptive Learning Optimizer.

Adam with a learnable brain that reads the loss landscape and reshapes
the update rule every step.  Each parameter tensor gets its own policy
(phi) and landscape signal (rho), so layers can adapt independently.
"""

import torch
from torch.optim import Optimizer

from .diagnostics import DiagnosticsTracker
from .meta_grads import compute_meta_grads


_PHI_BIAS = (1.4, 3.0, -2.0)

META_GRAD_CLIP = 1.0


def _build_phi_init(degree: int) -> torch.Tensor:
    """Build the initial phi vector for a given polynomial degree.

    Layout: three consecutive blocks of (degree+1) coefficients, one per
    policy variable (pm, pv, ps). Within each block the coefficients are
    ordered highest-power-first: [coeff_d, ..., coeff_1, coeff_0].

    Higher-order coefficients start at zero; the degree-0 (bias) coefficient
    gets the Adam-like default.
    """
    n = degree + 1
    phi = torch.zeros(3 * n)
    for i, bias in enumerate(_PHI_BIAS):
        phi[i * n + n - 1] = bias
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
    r"""HALO optimizer with per-parameter response policies.

    Each parameter tensor maintains its own landscape signal (rho) and
    policy vector (phi), allowing different layers to adapt independently.

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

        self._phi_init = _build_phi_init(degree)

        # param id -> human-readable name (populated by set_param_names)
        self._param_names: dict[int, str] = {}

        for group in self.param_groups:
            group["step"] = 0

    def set_param_names(self, named_parameters):
        """Register human-readable names for diagnostics.

        Call with model.named_parameters() after construction.
        """
        for name, p in named_parameters:
            self._param_names[id(p)] = name

    def _param_name(self, p: torch.Tensor) -> str:
        return self._param_names.get(id(p), f"param_{id(p)}")

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
            degree = group["degree"]
            n_phi = degree + 1

            group["step"] += 1
            step = group["step"]

            bias_corr1 = 1 - beta1 ** step
            bias_corr2 = 1 - beta2 ** step

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["g_prev"] = torch.zeros_like(p)
                    state["intermediates"] = None
                    state["phi"] = self._phi_init.clone().to(device=p.device)
                    state["rho"] = torch.zeros((), device=p.device)

                phi = state["phi"]
                rho = state["rho"]
                g_prev = state["g_prev"]

                # ---- Per-param landscape signal ----
                dot_val = (g * g_prev).sum()
                sq_g = g.pow(2).sum()
                sq_prev = g_prev.pow(2).sum()
                denom_cos = (sq_g * sq_prev).sqrt()
                r = torch.where(
                    denom_cos > 1e-12,
                    dot_val / denom_cos.clamp(min=1e-12),
                    torch.zeros((), device=p.device),
                )
                rho = gamma * rho + (1 - gamma) * r

                # ---- Per-param response policy ----
                coeffs_m = phi[:n_phi]
                coeffs_v = phi[n_phi : 2 * n_phi]
                coeffs_s = phi[2 * n_phi :]

                z_m = _horner(coeffs_m, rho)
                z_v = _horner(coeffs_v, rho)
                z_s = _horner(coeffs_s, rho)

                pm = torch.sigmoid(z_m)
                pv = 0.5 * torch.sigmoid(z_v)
                ps = torch.sigmoid(z_s)

                # ---- Meta-gradient + phi update ----
                intermediates = state["intermediates"]
                if intermediates is not None:
                    mg = compute_meta_grads(
                        g, intermediates, g_prev, phi, lr, degree=degree
                    )
                    mg_norm = mg.norm()
                    scale = torch.clamp(
                        META_GRAD_CLIP / (mg_norm + 1e-12), max=1.0
                    )
                    phi.add_(mg * scale, alpha=-eta_phi)

                # ---- Adam bookkeeping ----
                m = state["m"]
                v = state["v"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / bias_corr1
                v_hat = v / bias_corr2

                # ---- Adaptive step ----
                n = pm * m_hat + (1 - pm) * g
                n_abs = torch.clamp(n.abs(), min=1e-12)
                denom = v_hat.pow(pv) + eps
                d = -lr * n_abs.pow(1 - ps) * torch.sign(n) / denom

                if weight_decay != 0:
                    d = d - lr * weight_decay * p.data

                p.data.add_(d)

                # ---- Store intermediates for next step ----
                state["intermediates"] = {
                    "n_t": n.detach(),
                    "m_hat": m_hat.detach(),
                    "v_hat": v_hat.detach(),
                    "p_m": pm.detach(),
                    "p_v": pv.detach(),
                    "p_s": ps.detach(),
                    "rho": rho.detach().clone(),
                }

                g_prev.copy_(g)
                state["rho"] = rho

                # ---- Per-param diagnostics ----
                if self.diagnostics is not None:
                    self.diagnostics.record(
                        step,
                        self._param_name(p),
                        float(r),
                        float(rho),
                        float(pm),
                        float(pv),
                        float(ps),
                        phi,
                    )

        return loss
