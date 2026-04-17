import torch
from halo.meta_grads import compute_meta_grads
from halo.optimizer import _build_phi_init, _horner


def _make_intermediates(shape=(20,), device="cpu", degree=1):
    """Create a fake intermediates dict plus the g_t and phi arguments."""
    g = torch.randn(shape, device=device)
    m = torch.randn(shape, device=device)
    v = torch.rand(shape, device=device) + 0.1
    phi = _build_phi_init(degree).to(device=device)
    # Give the slope/higher-order coefficients small nonzero values for testing
    phi = phi + 0.1 * torch.randn_like(phi)
    rho = torch.tensor(0.3, device=device)

    n_phi = degree + 1
    coeffs_m = phi[:n_phi]
    coeffs_v = phi[n_phi : 2 * n_phi]
    coeffs_s = phi[2 * n_phi :]

    pm = torch.sigmoid(_horner(coeffs_m, rho))
    pv = 0.5 * torch.sigmoid(_horner(coeffs_v, rho))
    ps = torch.sigmoid(_horner(coeffs_s, rho))

    n = pm * m + (1 - pm) * g
    v_hat = v

    intermediates = {
        "n_t": n,
        "m_hat": m,
        "v_hat": v_hat,
        "p_m": pm,
        "p_v": pv,
        "p_s": ps,
        "rho": rho,
    }
    return intermediates, g, phi


class TestMetaGradientShape:
    def test_returns_correct_shape_degree1(self):
        intermediates, g_t, phi = _make_intermediates(degree=1)
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3, degree=1)
        assert result.shape == (6,)

    def test_returns_correct_shape_degree2(self):
        intermediates, g_t, phi = _make_intermediates(degree=2)
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3, degree=2)
        assert result.shape == (9,)

    def test_returns_correct_shape_degree3(self):
        intermediates, g_t, phi = _make_intermediates(degree=3)
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3, degree=3)
        assert result.shape == (12,)

    def test_returns_scalar_values(self):
        intermediates, g_t, phi = _make_intermediates()
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        for val in result:
            assert val.ndim == 0


class TestFiniteDifference:
    """Verify analytic meta-gradients against numerical finite differences."""

    def _compute_d(self, intermediates, eta):
        n = intermediates["n_t"]
        v_hat = intermediates["v_hat"]
        ps = intermediates["p_s"]
        pv = intermediates["p_v"]
        n_abs = torch.clamp(n.abs(), min=1e-12)
        denom = v_hat.pow(pv) + 1e-8
        return -eta * n_abs.pow(1 - ps) * torch.sign(n) / denom

    def _numerical_dd_dpm(self, intermediates, g_t, g_next, eta, eps=1e-5):
        d_t = self._compute_d(intermediates, eta)
        loss_ref = torch.sum(g_next * d_t)

        pm_orig = intermediates["p_m"]
        intermediates["p_m"] = pm_orig + eps
        intermediates["n_t"] = intermediates["p_m"] * intermediates["m_hat"] + (1 - intermediates["p_m"]) * g_t
        d_t_plus = self._compute_d(intermediates, eta)
        loss_plus = torch.sum(g_next * d_t_plus)
        intermediates["p_m"] = pm_orig
        intermediates["n_t"] = pm_orig * intermediates["m_hat"] + (1 - pm_orig) * g_t

        return (loss_plus - loss_ref) / eps

    def _numerical_dd_dpv(self, intermediates, g_next, eta, eps=1e-5):
        pv_orig = intermediates["p_v"]
        d_t = self._compute_d(intermediates, eta)
        loss_ref = torch.sum(g_next * d_t)

        intermediates["p_v"] = pv_orig + eps
        d_t_plus = self._compute_d(intermediates, eta)
        loss_plus = torch.sum(g_next * d_t_plus)
        intermediates["p_v"] = pv_orig

        return (loss_plus - loss_ref) / eps

    def _numerical_dd_dps(self, intermediates, g_next, eta, eps=1e-5):
        ps_orig = intermediates["p_s"]
        d_t = self._compute_d(intermediates, eta)
        loss_ref = torch.sum(g_next * d_t)

        intermediates["p_s"] = ps_orig + eps
        d_t_plus = self._compute_d(intermediates, eta)
        loss_plus = torch.sum(g_next * d_t_plus)
        intermediates["p_s"] = ps_orig

        return (loss_plus - loss_ref) / eps

    def test_dd_dpm_matches_finite_diff(self):
        torch.manual_seed(42)
        intermediates, g_t, phi = _make_intermediates()
        g_next = torch.randn(20)
        eta = 1e-3

        n_abs_neg_ps = torch.clamp(intermediates["n_t"].abs(), min=1e-12).pow(-intermediates["p_s"])
        analytical = torch.sum(
            g_next * (-eta * (1 - intermediates["p_s"]) * n_abs_neg_ps * (intermediates["m_hat"] - g_t) / (intermediates["v_hat"].pow(intermediates["p_v"]) + 1e-8))
        )
        numerical = self._numerical_dd_dpm(intermediates, g_t, g_next, eta)

        assert torch.isclose(analytical, numerical, atol=1e-4, rtol=1e-2), (
            f"dd_dpm: analytical={analytical:.6f}, numerical={numerical:.6f}"
        )

    def test_dd_dpv_matches_finite_diff(self):
        torch.manual_seed(42)
        intermediates, g_t, phi = _make_intermediates()
        g_next = torch.randn(20)
        eta = 1e-3

        n_abs_1ps = torch.clamp(intermediates["n_t"].abs(), min=1e-12).pow(1 - intermediates["p_s"])
        v_pow = intermediates["v_hat"].pow(intermediates["p_v"])
        denom = v_pow + 1e-8
        analytical = torch.sum(
            g_next * (eta * n_abs_1ps * torch.sign(intermediates["n_t"]) * v_pow * torch.log(torch.clamp(intermediates["v_hat"].abs(), min=1e-12)) / (denom * denom))
        )
        numerical = self._numerical_dd_dpv(intermediates, g_next, eta)

        assert torch.isclose(analytical, numerical, atol=1e-4, rtol=1e-2), (
            f"dd_dpv: analytical={analytical:.6f}, numerical={numerical:.6f}"
        )

    def test_dd_dps_matches_finite_diff(self):
        torch.manual_seed(42)
        intermediates, g_t, phi = _make_intermediates()
        g_next = torch.randn(20)
        eta = 1e-3

        n_abs_1ps = torch.clamp(intermediates["n_t"].abs(), min=1e-12).pow(1 - intermediates["p_s"])
        denom = intermediates["v_hat"].pow(intermediates["p_v"]) + 1e-8
        analytical = torch.sum(
            g_next * (eta * n_abs_1ps * torch.sign(intermediates["n_t"]) * torch.log(torch.clamp(intermediates["n_t"].abs(), min=1e-12)) / denom)
        )
        numerical = self._numerical_dd_dps(intermediates, g_next, eta)

        assert torch.isclose(analytical, numerical, atol=1e-4, rtol=1e-2), (
            f"dd_dps: analytical={analytical:.6f}, numerical={numerical:.6f}"
        )

    def test_full_meta_grad_finite_diff_degree1(self):
        """End-to-end finite-diff check: perturb each phi element."""
        torch.manual_seed(42)
        intermediates_orig, g_t, phi = _make_intermediates(degree=1)
        g_next = torch.randn(20)
        eta = 1e-3

        analytical = compute_meta_grads(g_next, intermediates_orig, g_t, phi, eta, degree=1)

        eps = 1e-5
        for idx in range(phi.shape[0]):
            phi_plus = phi.clone()
            phi_plus[idx] += eps

            inter_plus = _rebuild_intermediates(phi_plus, intermediates_orig, g_t, degree=1)
            d_plus = self._compute_d(inter_plus, eta)

            phi_minus = phi.clone()
            phi_minus[idx] -= eps

            inter_minus = _rebuild_intermediates(phi_minus, intermediates_orig, g_t, degree=1)
            d_minus = self._compute_d(inter_minus, eta)

            numerical = torch.sum(g_next * (d_plus - d_minus)) / (2 * eps)
            assert torch.isclose(analytical[idx], numerical, atol=1e-3, rtol=5e-2), (
                f"phi[{idx}]: analytical={analytical[idx]:.6f}, numerical={numerical:.6f}"
            )

    def test_full_meta_grad_finite_diff_degree2(self):
        """End-to-end finite-diff check for degree-2 polynomial."""
        torch.manual_seed(42)
        intermediates_orig, g_t, phi = _make_intermediates(degree=2)
        g_next = torch.randn(20)
        eta = 1e-3

        analytical = compute_meta_grads(g_next, intermediates_orig, g_t, phi, eta, degree=2)

        eps = 1e-5
        for idx in range(phi.shape[0]):
            phi_plus = phi.clone()
            phi_plus[idx] += eps

            inter_plus = _rebuild_intermediates(phi_plus, intermediates_orig, g_t, degree=2)
            d_plus = self._compute_d(inter_plus, eta)

            phi_minus = phi.clone()
            phi_minus[idx] -= eps

            inter_minus = _rebuild_intermediates(phi_minus, intermediates_orig, g_t, degree=2)
            d_minus = self._compute_d(inter_minus, eta)

            numerical = torch.sum(g_next * (d_plus - d_minus)) / (2 * eps)
            assert torch.isclose(analytical[idx], numerical, atol=1e-3, rtol=5e-2), (
                f"phi[{idx}]: analytical={analytical[idx]:.6f}, numerical={numerical:.6f}"
            )


def _rebuild_intermediates(phi, orig_intermediates, g_t, degree):
    """Rebuild intermediates from a perturbed phi (for finite-diff testing)."""
    rho = orig_intermediates["rho"]
    m_hat = orig_intermediates["m_hat"]
    v_hat = orig_intermediates["v_hat"]

    n_phi = degree + 1
    coeffs_m = phi[:n_phi]
    coeffs_v = phi[n_phi : 2 * n_phi]
    coeffs_s = phi[2 * n_phi :]

    pm = torch.sigmoid(_horner(coeffs_m, rho))
    pv = 0.5 * torch.sigmoid(_horner(coeffs_v, rho))
    ps = torch.sigmoid(_horner(coeffs_s, rho))

    n_t = pm * m_hat + (1 - pm) * g_t

    return {
        "n_t": n_t,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "p_m": pm,
        "p_v": pv,
        "p_s": ps,
        "rho": rho,
    }


class TestEdgeCases:
    def test_zero_gradient(self):
        intermediates, g_t, phi = _make_intermediates()
        g_next = torch.zeros(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        assert torch.allclose(result, torch.zeros(6), atol=1e-8)

    def test_small_v_hat(self):
        intermediates, g_t, phi = _make_intermediates()
        intermediates["v_hat"] = torch.full((20,), 1e-10)
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_extreme_rho_positive(self):
        intermediates, g_t, phi = _make_intermediates()
        intermediates["rho"] = torch.tensor(0.99)
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        assert not torch.any(torch.isnan(result))

    def test_extreme_rho_negative(self):
        intermediates, g_t, phi = _make_intermediates()
        intermediates["rho"] = torch.tensor(-0.99)
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        assert not torch.any(torch.isnan(result))

    def test_extreme_rho_degree2(self):
        intermediates, g_t, phi = _make_intermediates(degree=2)
        intermediates["rho"] = torch.tensor(0.99)
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3, degree=2)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_device_consistency(self):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        intermediates, g_t, phi = _make_intermediates(device=device)
        g_next = torch.randn(20, device=device)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        assert result.device.type == device
