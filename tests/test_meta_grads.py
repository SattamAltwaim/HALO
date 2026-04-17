import torch
from halo.meta_grads import compute_meta_grads


def _make_intermediates(shape=(20,), device="cpu"):
    """Create a fake intermediates dict plus the g_t and phi arguments."""
    g = torch.randn(shape, device=device)
    m = torch.randn(shape, device=device)
    v = torch.rand(shape, device=device) + 0.1
    phi = torch.tensor([0.0, 1.4, 0.0, -0.4, 0.0, -2.0], device=device)
    rho = torch.tensor(0.3, device=device)

    pm = torch.sigmoid(phi[0] * rho + phi[1])
    pv = 0.5 * torch.sigmoid(phi[2] * rho + phi[3])
    ps = torch.sigmoid(phi[4] * rho + phi[5])

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
    def test_returns_correct_shape(self):
        intermediates, g_t, phi = _make_intermediates()
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        assert result.shape == (6,)

    def test_returns_scalar_values(self):
        intermediates, g_t, phi = _make_intermediates()
        g_next = torch.randn(20)
        result = compute_meta_grads(g_next, intermediates, g_t, phi, eta=1e-3)
        for val in result:
            assert val.ndim == 0


class TestFiniteDifference:
    """Verify analytic meta-gradients against numerical finite differences."""

    def _numerical_dd_dpm(self, intermediates, g_t, g_next, eta, eps=1e-5):
        """Numerical approximation of dL/dp_m via finite differences."""
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

    def _compute_d(self, intermediates, eta):
        n = intermediates["n_t"]
        v_hat = intermediates["v_hat"]
        ps = intermediates["p_s"]
        pv = intermediates["p_v"]
        n_abs = torch.clamp(n.abs(), min=1e-12)
        denom = v_hat.pow(pv) + 1e-8
        return -eta * n_abs.pow(1 - ps) * torch.sign(n) / denom

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
