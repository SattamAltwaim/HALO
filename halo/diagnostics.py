"""
Diagnostics tracker for the HALO optimizer.

Per-parameter recording: each parameter tensor gets its own history of
rho, policy values, and phi coefficients.  Zero overhead when disabled.
"""


class DiagnosticsTracker:
    """Stores per-step, per-parameter optimizer internals for analysis."""

    def __init__(self, *, degree: int = 1):
        self._degree = degree
        self._n_phi = 3 * (degree + 1)
        self._phi_keys = [f"phi_{i}" for i in range(self._n_phi)]
        self._history: dict[str, dict] = {}
        self._step_set: set[int] = set()
        self._steps: list[int] = []

    def _ensure_param(self, param_name: str) -> dict:
        if param_name not in self._history:
            entry: dict[str, list] = {
                "r": [], "rho": [], "p_m": [], "p_v": [], "p_s": [],
            }
            for key in self._phi_keys:
                entry[key] = []
            self._history[param_name] = entry
        return self._history[param_name]

    def record(self, step, param_name, r, rho, pm, pv, ps, phi):
        """Record diagnostics for one parameter at one step."""
        if step not in self._step_set:
            self._step_set.add(step)
            self._steps.append(step)

        entry = self._ensure_param(param_name)
        entry["r"].append(float(r))
        entry["rho"].append(float(rho))
        entry["p_m"].append(float(pm))
        entry["p_v"].append(float(pv))
        entry["p_s"].append(float(ps))

        phi_vals = phi.detach().cpu().tolist()
        for i, key in enumerate(self._phi_keys):
            entry[key].append(phi_vals[i])

    def get_history(self):
        """Return full history: {"steps": [...], "params": {name: {...}}}."""
        return {
            "steps": list(self._steps),
            "params": {k: dict(v) for k, v in self._history.items()},
        }

    def param_names(self):
        """Return list of recorded parameter names in insertion order."""
        return list(self._history.keys())

    def summary(self, last_n=5):
        """Print a summary of the last N recorded steps per param."""
        if not self._steps:
            return "No data recorded."
        n = min(last_n, len(self._steps))
        lines = [f"Last {n} steps:"]
        for pname, entry in self._history.items():
            lines.append(f"  [{pname}]")
            for i in range(-n, 0):
                idx = len(entry["rho"]) + i
                if idx < 0:
                    continue
                lines.append(
                    f"    rho={entry['rho'][idx]:.4f} "
                    f"pm={entry['p_m'][idx]:.4f} pv={entry['p_v'][idx]:.4f} "
                    f"ps={entry['p_s'][idx]:.4f}"
                )
        return "\n".join(lines)
