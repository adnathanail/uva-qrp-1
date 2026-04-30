# A simple method for sampling random Clifford operators
# https://arxiv.org/abs/2008.06011
# Ewout van den Berg

# How to efficiently select an arbitrary Clifford group element
# Robert Koenig, John A. Smolin
# https://arxiv.org/abs/1406.2170

# uv run scripts/02_how_many_n_qubit_cliffords.py        # print + plot
# uv run scripts/02_how_many_n_qubit_cliffords.py plot   # plot only

from __future__ import annotations

import math
import sys
from pathlib import Path

N_MAX = 10
PLOT_FILE = Path("results/clifford_group_sizes.png")


def clifford_group_size(n):
    result = 2 ** (n**2 + 2 * n)
    for k in range(1, n + 1):
        result *= 4**k - 1
    return result


def log10_int(n: int) -> float:
    # math.log10 on a Python int converts via float, which overflows past ~1e308.
    # The number of digits gives the integer part of log10; the leading digits give the fraction.
    s = str(n)
    head = s[:15]
    return (len(s) - len(head)) + math.log10(int(head))


def plot_clifford_group_sizes(n_max: int = N_MAX, out: Path = PLOT_FILE) -> None:
    import matplotlib.pyplot as plt

    ns = list(range(1, n_max + 1))
    sizes = [clifford_group_size(n) for n in ns]
    log_sizes = [log10_int(s) for s in sizes]

    # Asymptotic reference: |C_n| ~ 2^(2n² + 3n) up to a constant from prod (1 - 4^-k).
    asymptotic = [(2 * n**2 + 3 * n) * math.log10(2) for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, log_sizes, "o-", label=r"$\log_{10}|\operatorname{Cl}_n|$", color="C0")
    ax.plot(ns, asymptotic, "--", label=r"$\log_{10}\, 2^{2n^2 + 3n}$", color="C1", alpha=0.7)

    for n, s, ls in zip(ns, sizes, log_sizes, strict=False):
        label = f"{s:_}" if n <= 4 else f"~{s:.1e}"
        ax.annotate(label, (n, ls), textcoords="offset points", xytext=(6, -4), fontsize=7)

    ax.set_xlabel("n (qubits)")
    ax.set_ylabel(r"$\log_{10}|\operatorname{Cl}_n|$  (≈ number of digits)")
    ax.set_title("Size of the n-qubit Clifford group")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    plot_only = len(sys.argv) > 1 and sys.argv[1] == "plot"

    if not plot_only:
        for i in range(1, N_MAX + 1):
            print(f"There are {clifford_group_size(i)} {i}-qubit Cliffords")

    plot_clifford_group_sizes()
