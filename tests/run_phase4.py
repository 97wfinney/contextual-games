"""
run_phase4.py -- Phase 4 production run.

For every class in data/catalogue_quantum.csv, re-derive the optimal
qubit strategy, build its empirical model, and compute:

  Q2 columns:  ncf, cf, hardness, bound (= ncf * hardness), p_F,
               slack (= p_F - bound; ~0 means Theorem 4 is tight here);
  Q3 columns:  eta_adv (noise killing the advantage, closed form),
               eta_cf (noise killing the resource, bisection),
               eta_gap (= eta_cf - eta_adv).

Writes data/analysis_phase4.csv and prints the two headline tables.

Interpretation crib:
  slack = 0     the strategy's failures are exactly its classical
                fraction failing at the unavoidable rate; nothing wasted.
  slack > 0     the bound is loose for this strategy/game: either the
                classical part fails on more than the minimum-violation
                constraints, or the contextual part also fails sometimes.
  eta_gap = 0   advantage and resource die at the same noise: the
                witnessing Bell inequality is (a rescaling of) the game
                functional.
  eta_gap > 0   there is a noise band where the strategy is still
                contextual but no longer beats the best classical score:
                contextuality present, advantage gone -- resource without
                usefulness for THIS game.

Use --family to run a subset (xor-2party | xor-3party | general-2party).
"""

import argparse
import csv
import os
import sys
import time

from contextual_games.enumeration import bits_from_id
from contextual_games.analysis import (
    optimal_strategy_model, tightness_row,
    advantage_threshold, cf_threshold,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CATALOGUE = os.path.join(DATA_DIR, "catalogue_quantum.csv")
OUT = os.path.join(DATA_DIR, "analysis_phase4.csv")


def analyse(entry, restarts=8, seed=0):
    n = 2 if entry["family"].endswith("2party") else 3
    cid = int(entry["canonical_id"])
    bits = bits_from_id(cid, 4 ** n)
    value, model = optimal_strategy_model(bits, n, restarts=restarts,
                                          seed=seed)
    row = {
        "canonical_id": cid,
        "family": entry["family"],
        "omega_c": float(entry["omega_c"]),
        "omega_q": value,
        "omega_ns": float(entry["omega_ns"]),
    }
    row.update(tightness_row(bits, n, model))
    row["eta_adv"] = advantage_threshold(bits, n, model)
    row["eta_cf"] = cf_threshold(model)
    row["eta_gap"] = row["eta_cf"] - row["eta_adv"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    with open(CATALOGUE) as f:
        entries = list(csv.DictReader(f))
    if args.family:
        entries = [e for e in entries if e["family"] == args.family]

    rows = []
    for i, e in enumerate(entries):
        t0 = time.time()
        row = analyse(e)
        rows.append(row)
        print(f"  {i + 1}/{len(entries)} {row['family']} "
              f"id={row['canonical_id']}  slack={row['slack']:.6f}  "
              f"eta_adv={row['eta_adv']:.4f}  eta_cf={row['eta_cf']:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    mode = "a" if args.append and os.path.exists(OUT) else "w"
    with open(OUT, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({mode})")

    print("\nQ2 -- Theorem 4 tightness:")
    print(f"  {'family':>15} {'id':>22} {'cf':>7} {'bound':>7} "
          f"{'p_F':>7} {'slack':>8}")
    for r in sorted(rows, key=lambda r: -r["slack"]):
        print(f"  {r['family']:>15} {r['canonical_id']:>22} "
              f"{r['cf']:>7.4f} {r['bound']:>7.4f} {r['p_F']:>7.4f} "
              f"{r['slack']:>8.5f}")

    print("\nQ3 -- noise thresholds:")
    print(f"  {'family':>15} {'id':>22} {'eta_adv':>8} {'eta_cf':>8} "
          f"{'eta_gap':>8}")
    for r in sorted(rows, key=lambda r: -r["eta_gap"]):
        print(f"  {r['family']:>15} {r['canonical_id']:>22} "
              f"{r['eta_adv']:>8.4f} {r['eta_cf']:>8.4f} "
              f"{r['eta_gap']:>8.4f}")


if __name__ == "__main__":
    main()
