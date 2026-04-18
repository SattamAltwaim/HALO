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
        assert opt.defaults["degree"] == 1
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

    def test_invalid_degree_zero(self):
        p = torch.zeros(10, requires_grad=True)
        with pytest.raises(ValueError):
            HALO([p], degree=0)

    def test_invalid_degree_float(self):
        p = torch.zeros(10, requires_grad=True)
        with pytest.raises(ValueError):
            HALO([p], degree=1.5)

    def test_invalid_degree_negative(self):
        p = torch.zeros(10, requires_grad=True)
        with pytest.raises(ValueError):
            HALO([p], degree=-1)

    def test_phi_size_degree1(self):
        p = torch.zeros(10, requires_grad=True)
        opt = HALO([p], degree=1)
        # phi is lazily created in state on first step
        p.grad = torch.randn(10)
        opt.step()
        assert opt.state[p]["phi"].shape == (6,)

    def test_phi_size_degree2(self):
        p = torch.zeros(10, requires_grad=True)
        opt = HALO([p], degree=2)
        p.grad = torch.randn(10)
        opt.step()
        assert opt.state[p]["phi"].shape == (9,)

    def test_phi_size_degree3(self):
        p = torch.zeros(10, requires_grad=True)
        opt = HALO([p], degree=3)
        p.grad = torch.randn(10)
        opt.step()
        assert opt.state[p]["phi"].shape == (12,)


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

        # Run one step to lazily init phi
        opt.zero_grad()
        _forward(W, x, target).backward()
        opt.step()
        phi_init = opt.state[W]["phi"].clone()

        for _ in range(19):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        phi_final = opt.state[W]["phi"]
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
        assert len(history["steps"]) == 5
        pnames = list(history["params"].keys())
        assert len(pnames) == 1
        entry = history["params"][pnames[0]]
        assert len(entry["p_m"]) == 5
        assert len(entry["rho"]) == 5
        assert "phi_0" in entry

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
        # Trigger lazy init
        p.grad = torch.randn(10)
        opt.step()
        phi = opt.state[p]["phi"]

        a1, b1, a2, b2, a3, b3 = phi.unbind()
        rho = torch.tensor(0.0)

        pm = torch.sigmoid(a1 * rho + b1)
        pv = 0.5 * torch.sigmoid(a2 * rho + b2)
        ps = torch.sigmoid(a3 * rho + b3)

        assert 0.7 < pm < 0.9
        assert 0.4 < pv < 0.5
        assert 0.05 < ps < 0.2

    def test_rho_near_zero_policy_degree2(self):
        """Higher-degree phi with zero higher-order coeffs gives same result."""
        p = torch.randn(10, requires_grad=True)
        opt = HALO([p], lr=1e-3, degree=2)
        p.grad = torch.randn(10)
        opt.step()
        phi = opt.state[p]["phi"]
        rho = torch.tensor(0.0)

        n = 3  # degree + 1
        from halo.optimizer import _horner
        pm = torch.sigmoid(_horner(phi[:n], rho))
        pv = 0.5 * torch.sigmoid(_horner(phi[n : 2 * n], rho))
        ps = torch.sigmoid(_horner(phi[2 * n :], rho))

        assert 0.7 < pm < 0.9
        assert 0.4 < pv < 0.5
        assert 0.05 < ps < 0.2


class TestPolynomialDegree:
    def test_degree2_converges(self):
        """Degree-2 optimizer should still converge on the quadratic problem."""
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1, degree=2)
        loss0 = _forward(W, x, target)

        for _ in range(100):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        final_loss = _forward(W, x, target)
        assert final_loss < loss0 * 0.1

    def test_degree3_converges(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1, degree=3)
        loss0 = _forward(W, x, target)

        for _ in range(100):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        final_loss = _forward(W, x, target)
        assert final_loss < loss0 * 0.1

    def test_degree2_phi_updates(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1, eta_phi=0.1, degree=2)

        opt.zero_grad()
        _forward(W, x, target).backward()
        opt.step()
        phi_init = opt.state[W]["phi"].clone()

        for _ in range(19):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        phi_final = opt.state[W]["phi"]
        assert not torch.allclose(phi_init, phi_final)

    def test_degree2_diagnostics(self):
        W, x, target = _make_quadratic_problem()
        opt = HALO([W], lr=0.1, degree=2, diagnostics=True)

        for _ in range(5):
            opt.zero_grad()
            loss = _forward(W, x, target)
            loss.backward()
            opt.step()

        history = opt.diagnostics.get_history()
        assert len(history["steps"]) == 5
        pnames = list(history["params"].keys())
        assert len(pnames) == 1
        entry = history["params"][pnames[0]]
        for i in range(9):
            assert f"phi_{i}" in entry
            assert len(entry[f"phi_{i}"]) == 5

    def test_degree1_backward_compatible_phi_init(self):
        """Degree-1 phi_init should match the original hardcoded values."""
        from halo.optimizer import _build_phi_init
        phi = _build_phi_init(1)
        expected = torch.tensor([0.0, 1.4, 0.0, 3.0, 0.0, -2.0])
        assert torch.allclose(phi, expected)


class TestPerParamPolicy:
    """Tests specific to the per-parameter policy design."""

    def test_each_param_has_own_phi(self):
        W1 = torch.randn(5, 5, requires_grad=True)
        W2 = torch.randn(5, 5, requires_grad=True)
        opt = HALO([W1, W2], lr=0.1, eta_phi=0.1)

        for _ in range(20):
            opt.zero_grad()
            loss = (W1.sum() + W2.pow(2).sum())
            loss.backward()
            opt.step()

        phi1 = opt.state[W1]["phi"]
        phi2 = opt.state[W2]["phi"]
        assert phi1.shape == (6,)
        assert phi2.shape == (6,)
        assert not torch.allclose(phi1, phi2)

    def test_each_param_has_own_rho(self):
        W1 = torch.randn(5, 5, requires_grad=True)
        W2 = torch.randn(5, 5, requires_grad=True)
        opt = HALO([W1, W2], lr=0.1)

        for _ in range(10):
            opt.zero_grad()
            loss = (W1.sum() + W2.pow(2).sum())
            loss.backward()
            opt.step()

        rho1 = opt.state[W1]["rho"]
        rho2 = opt.state[W2]["rho"]
        assert rho1.shape == ()
        assert rho2.shape == ()

    def test_set_param_names(self):
        import torch.nn as nn
        model = nn.Linear(5, 3)
        opt = HALO(model.parameters(), lr=0.1, diagnostics=True)
        opt.set_param_names(model.named_parameters())

        x = torch.randn(2, 5)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        history = opt.diagnostics.get_history()
        pnames = list(history["params"].keys())
        assert "weight" in pnames
        assert "bias" in pnames

    def test_multi_param_diagnostics(self):
        W1 = torch.randn(5, 5, requires_grad=True)
        W2 = torch.randn(3, 3, requires_grad=True)
        opt = HALO([W1, W2], lr=0.1, diagnostics=True)

        for _ in range(5):
            opt.zero_grad()
            (W1.sum() + W2.sum()).backward()
            opt.step()

        history = opt.diagnostics.get_history()
        assert len(history["steps"]) == 5
        assert len(history["params"]) == 2
        for entry in history["params"].values():
            assert len(entry["p_m"]) == 5
