# uv run algorithm-1/scripts/freeze_new_stim_clifford.py 4
import argparse

from lib.unitaries.generators import freeze_stim_clifford

parser = argparse.ArgumentParser(description="Freeze a random Stim Clifford unitary.")
parser.add_argument("n", type=int, help="Number of qubits")
args = parser.parse_args()

name = freeze_stim_clifford(args.n)
print(f"Frozen: {name}")
