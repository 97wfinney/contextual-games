"""
compute_quantum_values.py -- Phase 3 production run.

Adds quantum brackets to the interesting corner of the Phase 2 catalogue:

  * both n=2 XOR classes,
  * all five n=3 XOR classes,
  * every general 2-party class with omega_ns > omega_c (there are 9).

Writes data/catalogue_quantum.csv with columns:
  canonical_id, family, omega_c, omega_ns, omega_q_lower, omega_q_upper,
  certified, qc_gap (= omega_q_lower - omega_c, the certified quantum
  advantage), nsq_gap (= omega_ns - omega_q_upper, the certified
  super-quantum gap).

For bipartite XOR classes the bracket is exact (Tsirelson). Elsewhere,
certified=False rows must be reported as intervals.
"""

import csv
import os
import time

from contextual_games.enumeration import (
    bell_scenario, bits_from_id, xor_game_bits, canonical_form,
    symmetry_group, enumerate_two_party_classes, enumerate_xor_classes,
    game_from_bits, solve_game_bits, build_catalogue,
)
from contextual_games.quantum import quantum_bracket

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def classify(label, n, class_ids, restarts, seed=0):
    sc = bell_scenario(n)
    n_cells = 4 ** n
    rows = []
    for i, cid in enumerate(class_ids):
        bits = bits_from_id(cid, n_cells)
        t0 = time.time()
        row = {"canonical_id": cid, "family": label}
        row.update(solve_game_bits(sc, bits))
        br = quantum_bracket(bits, n, restarts=restarts, seed=seed)
        row.update(br)
        row["qc_gap"] = max(0.0, br["omega_q_lower"] - row["omega_c"])
        row["nsq_gap"] = max(0.0, row["omega_ns"] - br["omega_q_upper"])
        rows.append(row)
        print(f"  {label} {i + 1}/{len(class_ids)}  id={cid}  "
              f"wc={row['omega_c']:.4f}  "
              f"wq=[{br['omega_q_lower']:.4f},{br['omega_q_upper']:.4f}]"
              f"{'*' if br['certified'] else ' '}  "
              f"wns={row['omega_ns']:.4f}   ({time.time() - t0:.1f}s)")
    return rows


def main():
    rows = []

    print("n=2 XOR classes:")
    xor2 = enumerate_xor_classes(2)
    rows += classify("xor-2party", 2, sorted(xor2), restarts=6)

    print("n=3 XOR classes:")
    xor3 = enumerate_xor_classes(3)
    rows += classify("xor-3party", 3, sorted(xor3), restarts=10)

    print("general 2-party classes with NS advantage:")
    classes2 = enumerate_two_party_classes()
    sc2 = bell_scenario(2)
    interesting = []
    for cid in sorted(classes2):
        r = solve_game_bits(sc2, bits_from_id(cid, 16))
        if r["ns_gap"] > 1e-9:
            interesting.append(cid)
    rows += classify("general-2party", 2, interesting, restarts=8)

    path = build_catalogue(rows, os.path.join(
        DATA_DIR, "catalogue_quantum.csv"))
    print(f"\nwrote {path}")

    print("\nquantum advantage summary (certified rows marked *):")
    print(f"  {'family':>15} {'id':>22} {'wc':>7} {'wq':>17} "
          f"{'wns':>7} {'qc_gap':>7}")
    for r in sorted(rows, key=lambda r: -r["qc_gap"]):
        wq = (f"{r['omega_q_lower']:.4f}" if r["certified"]
              else f"[{r['omega_q_lower']:.3f},{r['omega_q_upper']:.3f}]")
        print(f"  {r['family']:>15} {r['canonical_id']:>22} "
              f"{r['omega_c']:>7.4f} {wq:>17}"
              f"{'*' if r['certified'] else ' '} "
              f"{r['omega_ns']:>7.4f} {r['qc_gap']:>7.4f}")


if __name__ == "__main__":
    main()
