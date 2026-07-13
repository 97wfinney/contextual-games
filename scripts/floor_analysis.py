"""
floor_analysis.py -- Experiment 1: the contextuality floor.

The strengthened bound, rearranged, is a floor on the contextual
fraction of ANY strategy:

    CF(e)  >=  (p_S(e) - omega_c) / (omega_ns - omega_c)

(proved: two lines from the ABM decomposition theorem). At a quantum
optimum, p_S = omega_q, so every optimal strategy satisfies

    CF(e*)  >=  floor  :=  (omega_q - omega_c) / (omega_ns - omega_c).

This script reads data/analysis_phase4.csv (which already records
omega_c, omega_q, omega_ns, and the cf of the located optimum per
class), computes the floor and the EXCESS = cf - floor, and writes
data/floor_analysis.csv.

Reading the excess column:
    excess ~ 0      the located optimum sits ON the proved floor:
                    minimally contextual among all strategies achieving
                    this score (saturation).
    excess > 0      the located optimum carries more contextuality than
                    the floor requires. Either the true min over the
                    optimal set is above the floor (a genuine
                    achievability failure -- interesting), or the search
                    merely landed on a non-minimal member of the optimal
                    set (test with scripts/multiseed_cf.py).
    excess < -tol   impossible: would violate a theorem. Treat as a bug.

Games with omega_ns = omega_c (no post-classical structure) have an
undefined floor 0/0; they are reported with floor 0 and flagged.
"""

import csv
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TOL = 1e-6


def main():
    with open(os.path.join(DATA, "analysis_phase4.csv")) as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        wc, wq, wns = (float(r["omega_c"]), float(r["omega_q"]),
                       float(r["omega_ns"]))
        cf = float(r["cf"])
        gap = wns - wc
        degenerate = gap < TOL
        floor = 0.0 if degenerate else (wq - wc) / gap
        floor = min(max(floor, 0.0), 1.0)
        excess = cf - floor
        out.append({
            "canonical_id": r["canonical_id"],
            "family": r["family"],
            "omega_c": wc, "omega_q": wq, "omega_ns": wns,
            "cf_located_optimum": cf,
            "cf_floor": floor,
            "excess": excess,
            "on_floor": abs(excess) < 1e-4,
            "degenerate": degenerate,
        })
        if excess < -1e-4:
            raise AssertionError(
                f"class {r['canonical_id']}: cf {cf} below proved floor "
                f"{floor} -- theorem violated, investigate immediately")

    path = os.path.join(DATA, "floor_analysis.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{'family':>15} {'id':>22} {'floor':>8} {'cf found':>9} "
          f"{'excess':>8}  on floor?")
    for r in sorted(out, key=lambda r: -r["excess"]):
        flag = "degenerate" if r["degenerate"] else (
            "YES" if r["on_floor"] else "no")
        print(f"{r['family']:>15} {r['canonical_id']:>22} "
              f"{r['cf_floor']:>8.4f} {r['cf_located_optimum']:>9.4f} "
              f"{r['excess']:>8.4f}  {flag}")

    n_on = sum(1 for r in out if r["on_floor"] and not r["degenerate"])
    n_int = sum(1 for r in out if not r["degenerate"])
    print(f"\nnon-degenerate classes on the floor: {n_on} / {n_int}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()