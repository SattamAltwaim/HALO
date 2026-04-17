"""
Diagnostics tracker for the HALO optimizer.

Only records data when explicitly enabled. Zero overhead when disabled.
"""


class DiagnosticsTracker:
    """Stores per-step optimizer internals for analysis."""

    def __init__(self, *, degree: int = 1):
        self._degree = degree
        n_phi = 3 * (degree + 1)
        self._phi_keys = [f"phi_{i}" for i in range(n_phi)]
        self._history = {
            "step": [],
            "r": [],
            "rho": [],
            "p_m": [],
            "p_v": [],
            "p_s": [],
        }
        for key in self._phi_keys:
            self._history[key] = []

    def record(self, step, r, rho, pm, pv, ps, phi):
        """Record diagnostics for one step."""
        self._history["step"].append(step)
        self._history["r"].append(float(r))
        self._history["rho"].append(float(rho))
        self._history["p_m"].append(float(pm))
        self._history["p_v"].append(float(pv))
        self._history["p_s"].append(float(ps))
        phi_vals = phi.detach().cpu().tolist()
        for i, key in enumerate(self._phi_keys):
            self._history[key].append(phi_vals[i])

    def get_history(self):
        """Return full history as dict of lists."""
        return dict(self._history)

    def summary(self, last_n=5):
        """Print a summary of the last N recorded steps."""
        h = self._history
        if not h["step"]:
            return "No data recorded."
        lines = [f"Last {min(last_n, len(h['step']))} steps:"]
        for i in range(-last_n, 0):
            idx = len(h["step"]) + i
            lines.append(
                f"  step={h['step'][idx]} r={h['r'][idx]:.4f} rho={h['rho'][idx]:.4f} "
                f"pm={h['p_m'][idx]:.4f} pv={h['p_v'][idx]:.4f} ps={h['p_s'][idx]:.4f}"
            )
        return "\n".join(lines)
