"""Executes analysis code written by the model and captures its output and charts."""
import contextlib
import io
import re
import threading
import traceback
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

MAX_OUTPUT_CHARS = 4000
EXEC_TIMEOUT_S = 60

# Show full-width tables so the model sees every column instead of "...".
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

# One consistent, clean chart style for everything the model draws.
CHART_COLORS = ["#76B900", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6", "#14B8A6", "#EC4899", "#64748B"]
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 110,
    "axes.prop_cycle": cycler(color=CHART_COLORS),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.frameon": False,
})

# Model-written code runs in-process, so stop the obvious ways it could touch the host:
# shell/OS access, reading files (e.g. .env), or leaking API keys. This is a guard rail against
# accidents and casual prompt injection, not a hardened security boundary.
_MODULES = r"(os|sys|subprocess|shutil|socket|pathlib|requests|urllib|http|importlib|ctypes|pickle)"
BLOCKED_PATTERNS = [
    (rf"\bimport\s+{_MODULES}\b", "importing system modules"),
    (rf"\bfrom\s+{_MODULES}\b", "importing system modules"),
    (r"__import__|__builtins__|__subclasses__|__globals__", "dunder access"),
    (r"\bopen\s*\(", "opening files"),
    (r"(?<![\w.])(eval|exec|compile|globals|getattr)\s*\(", "dynamic code execution"),
    (r"\.read_\w+\s*\(|\bnp\.(load|loadtxt|genfromtxt|fromfile)\b", "reading files"),
    (r"environ|getenv|api_key|secret", "accessing credentials"),
]


def _humanize(value: float, _pos=None) -> str:
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"{value / divisor:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:g}"


def polish(fig) -> None:
    """Replaces '1e6'-style offset labels on large numeric axes with readable ones like 4.2M."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            # Date and categorical axes use other formatters and are left alone.
            if type(axis.get_major_formatter()) is ScalarFormatter and max(map(abs, axis.get_view_interval())) >= 1e4:
                axis.set_major_formatter(FuncFormatter(_humanize))


def find_blocked(code: str) -> str | None:
    for pattern, reason in BLOCKED_PATTERNS:
        if re.search(pattern, code, flags=re.IGNORECASE):
            return reason
    return None


@dataclass
class ExecResult:
    output: str
    error: str | None = None
    figures: list[bytes] = field(default_factory=list)


class Sandbox:
    """A persistent Python namespace (like a notebook kernel) holding the dataset as `df`."""

    def __init__(self, df: pd.DataFrame, web_search):
        self.sources: list[dict] = []

        def _web_search(query: str, max_results: int = 5) -> list[dict]:
            results = web_search(query, max_results)
            self.sources.extend(r for r in results if r.get("url") and r not in self.sources)
            return results

        self.namespace = {"df": df, "pd": pd, "np": np, "plt": plt, "web_search": _web_search}

    def run(self, code: str) -> ExecResult:
        blocked = find_blocked(code)
        if blocked:
            return ExecResult(output="", error=f"SecurityError: {blocked} is not allowed in the analysis sandbox. "
                                               "Work only with df and the provided tools.")

        plt.close("all")
        stdout = io.StringIO()
        errors: list[str] = []

        def target():
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(compile(code, "<analysis>", "exec"), self.namespace)
            except Exception:
                errors.append(traceback.format_exc(limit=3))

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(EXEC_TIMEOUT_S)
        if worker.is_alive():
            errors.append(f"TimeoutError: code ran longer than {EXEC_TIMEOUT_S}s. "
                          "Use a faster approach (vectorized pandas, aggregate first, or sample the data).")

        figures = []
        for num in plt.get_fignums():
            buf = io.BytesIO()
            fig = plt.figure(num)
            polish(fig)
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            figures.append(buf.getvalue())
        plt.close("all")

        output = stdout.getvalue()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        return ExecResult(output=output, error=errors[0] if errors else None, figures=figures)
