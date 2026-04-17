import pytest
import torch
from halo import HALO


def _make_quadratic_problem(dim=10, device="cpu"):
    """Simple quadratic loss: L = 0.5 * ||Wx||^2, minimum at W=0."""
    torch.manual_seed(42)
    x = torch.randn(dim, dim, device=device)
    target = torch.zeros(dim, dim, device=device)
    W = torch.randn(dim, dim, device=device, requires_grad=True)
    return W, x, target


def _forward(W, x, target):
    out = W @ x
    loss = 0.5 * torch.nn.functional.mse_loss(out, target)
    return loss


class TestInstantiation:
    def test_default_params(self):
        p = torch.zeros(10, requires_grad=True)
        opt = HALO([p])
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["weight_decay"] == 0.01
        assert opt.defaults["gamma"] == 0.99
        assert opt.defaults["eta_phi"] == 0.01
        assert opt.diagnostics is None

    def test_diagnostics_enabled(self):
        p = torch.zeros(10, requires_grad=True)
        opt = HALO([p], diagnostics=True)
        assert opt.diagnostics is not None

    def test_invalid_lr(self):
        p = torch.zeros(10, requires_grad=True)
        with pytest.raises(ValueError):
            HALO([p], lr=-0.1)

    def test_invalid_beta(self):
        p = torch.zeros(10, requires_grad=True)
        with pytest.raises(ValueError):
            HALO([p], betas=(1.5, 0.999))

    def test_invalid_gamma(self):
        p = torch.zeros(10, requires_grad=True)
        with pytest.raises(ValueError):
            HALO([p], gamma=1.5)


class TestBasicStep:
    def test_step_changes_params(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.01)

        loss0 = _forward(W, x, target)
        loss0.backward()
        opt.step()

        W.grad = None
        loss1 = _forward(W, x, target)
        assert loss1 < loss0 or torch.allclose(loss1, loss0, atol=1e-3)

    def test_multiple_steps(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1)
        loss0 = _forward(W, x, target)

        for _ in range(100):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        final_loss = _forward(W, x, target)
        assert final_loss < loss0 * 0.1

    def test_phi_updates(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1, eta_phi=0.1, diagnostics=True)

        phi_init = opt.param_groups[0]["phi"].clone()

        for _ in range(20):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        phi_final = opt.param_groups[0]["phi"]
        assert not torch.allclose(phi_init, phi_final)

    def test_diagnostics_records(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1, diagnostics=True)

        for _ in range(5):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        history = opt.diagnostics.get_history()
        assert len(history["step"]) == 5
        assert all(len(v) == 5 for v in history.values())
        assert "p_m" in history
        assert "rho" in history

    def test_no_diagnostics_no_overhead(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1, diagnostics=False)
        assert opt.diagnostics is None

        for _ in range(5):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

    def test_weight_decay(self):
        W, x, target = _make_quadratic_problem()
        opt_no_decay = HALO([W], lr=0.01, weight_decay=0.0)
        opt_decay = HALO([W.clone().detach().requires_grad_(True)], lr=0.01, weight_decay=0.5)

        for _ in range(5):
            for opt in [opt_no_decay, opt_decay]:
                opt.zero_grad()
                w = opt.param_groups[0]["params"][0]
                loss = _forward(w, x, target)
                loss.backward()
                opt.step()

        w0 = opt_no_decay.param_groups[0]["params"][0]
        w1 = opt_decay.param_groups[0]["params"][0]
        assert w1.norm() < w0.norm()


class TestMultiDevice:
    @pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="No GPU")
    def test_multi_device_step(self):
        device = "cuda" if torch.cuda.is_available() else "mps"
        W1 = torch.randn(5, 5, device=device, requires_grad=True)
        W2 = torch.randn(5, 5, device=device, requires_grad=True)
        opt = HALO([W1, W2], lr=0.01)

        x = torch.randn(5, 5, device=device)
        loss = (W1 @ x).sum() + (W2 @ x).sum()
        loss.backward()
        opt.step()
        assert W1.grad is not None


class TestResponsePolicy:
    def test_rho_near_zero_policy(self):
        """At rho=0, policy should produce near-Adam defaults."""
        p = torch.randn(10, requires_grad=True)
        opt = HALO([p], lr=1e-3)
        phi = opt.param_groups[0]["phi"]

        a1, b1, a2, b2, a3, b3 = phi.unbind()
        rho = torch.tensor(0.0)

        pm = torch.sigmoid(a1 * rho + b1)
        pv = 0.5 * torch.sigmoid(a2 * rho + b2)
        ps = torch.sigmoid(a3 * rho + b3)

        assert 0.7 < pm < 0.9
        assert 0.1 < pv < 0.4
        assert 0.05 < ps < 0.2
