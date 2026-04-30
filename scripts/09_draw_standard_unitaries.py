"""Draw each standard unitary's circuit using qiskit."""

from pathlib import Path

from cliff_lib.unitaries.standard import STANDARD_UNITARIES

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "standard_unitaries"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, build in STANDARD_UNITARIES.items():
        qc = build()
        fig = qc.draw(output="mpl", fold=-1)
        out_path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path.relative_to(OUTPUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
