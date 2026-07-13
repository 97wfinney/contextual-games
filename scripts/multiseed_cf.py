"""
multiseed_cf.py -- Experiment 2: is the excess contextuality real?

For each class with positive excess in data/floor_analysis.csv, this
script probes the OPTIMAL SET of the game:

  Stage 1 (diversify): many independent random-start searches for the
  quantum value; every angle-set achieving the certified value (within
  tolerance) is a genuine member of the optimal set. Compute the CF of
  each member found.

  Stage 2 (push down): starting from the lowest-CF member found, run a
  penalised minimisation of CF over angle space,

      minimise  CF(strategy(angles)) + K * max(0, target - value(angles)),

  which walks ALONG the optimal set (the penalty keeps the value pinned)
  looking for less contextual optima.

Interpretation of min-CF found, per class:
  -> falls to the floor:     the earlier excess was a search artefact;
                             the located optimum was just a non-minimal
                             member of the optimal set. Saturation holds.
  -> stalls above the floor: candidate genuine achievability failure --
                             the quantum optimal set may not reach the
                             NS-minimal contextuality at its score. The
                             most interesting outcome; escalate (more
                             seeds, qutrit ansatz, exact methods).

Writes data/multiseed_cf.csv. Runtime ~1-2 min for the six n=2 classes.
"""

import csv
import os

import numpy as np
from scipy.optimize import minimize

from contextual_games.enumeration import bits_from_id
from contextual_games.quantum import game_operator
from contextual_games.analysis import strategy_model
from contextual_games.cfraction import contextual_fraction

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
VALUE_TOL = 1e-6      # membership of the optimal set
PENALTY = 50.0        # stage-2 constraint weight
N_SEEDS = 15


def value(bits, n, angles):
    return float(np.linalg.eigvalsh(game_operator(bits, n, angles))[-1])


def cf_of(bits, n, angles):
    return contextual_fraction(strategy_model(bits, n, angles))


def probe_class(cid, n, target, floor, rng):
    bits = bits_from_id(int(cid), 4 ** n)
    members = []          # (cf, angles) of optimal-set members found

    # -- stage 1: diversify --------------------------------------------------
    for _ in range(N_SEEDS):
        x0 = np.stack([rng.uniform(0, np.pi, size=(n, 2)),
                       rng.uniform(0, 2 * np.pi, size=(n, 2))],
                      axis=-1).ravel()
        res = minimize(lambda a: -value(bits, n, a.reshape(n, 2, 2)),
                       x0, method="Nelder-Mead",
                       options={"maxiter": 1500, "fatol": 1e-12,
                                "xatol": 1e-8})
        ang = res.x.reshape(n, 2, 2)
        if value(bits, n, ang) >= target - VALUE_TOL:
            members.append((cf_of(bits, n, ang), ang))

    if not members:
        return {"n_optima": 0}

    cfs = sorted(m[0] for m in members)
    best_cf, best_ang = min(members, key=lambda m: m[0])

    # -- stage 2: minimise CF along the optimal set ---------------------------
    def penalised(flat):
        ang = flat.reshape(n, 2, 2)
        return (cf_of(bits, n, ang)
                + PENALTY * max(0.0, target - value(bits, n, ang)))

    res = minimize(penalised, best_ang.ravel(), method="Nelder-Mead",
                   options={"maxiter": 2000, "fatol": 1e-10,
                            "xatol": 1e-8})
    ang2 = res.x.reshape(n, 2, 2)
    v2 = value(bits, n, ang2)
    cf2 = cf_of(bits, n, ang2)
    refined_ok = v2 >= target - VALUE_TOL
    min_cf = min(best_cf, cf2) if refined_ok else best_cf

    return {
        "n_optima": len(members),
        "cf_max": cfs[-1], "cf_min_stage1": cfs[0],
        "cf_min_refined": min_cf,
        "refined_value": v2, "refined_kept_value": refined_ok,
        "gap_to_floor": min_cf - floor,
    }


def main():
    with open(os.path.join(DATA, "floor_analysis.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if float(r["excess"]) > 1e-4]
    print(f"probing {len(rows)} positive-excess classes, "
          f"{N_SEEDS} seeds each\n")

    rng = np.random.default_rng(2026)
    out = []
    for r in rows:
        n = 2 if r["family"].endswith("2party") else 3
        target = float(r["omega_q"])
        floor = float(r["cf_floor"])
        res = probe_class(r["canonical_id"], n, target, floor, rng)
        res.update({"canonical_id": r["canonical_id"],
                    "family": r["family"],
                    "cf_floor": floor,
                    "cf_originally_found": float(r["cf_located_optimum"])})
        out.append(res)
        print(f"  id={r['canonical_id']:>6}  floor={floor:.4f}  "
              f"optima found={res.get('n_optima', 0):>2}  "
              f"CF range=[{res.get('cf_min_stage1', float('nan')):.4f},"
              f"{res.get('cf_max', float('nan')):.4f}]  "
              f"min after refine={res.get('cf_min_refined', float('nan')):.4f}  "
              f"gap to floor={res.get('gap_to_floor', float('nan')):+.4f}",
              flush=True)

    path = os.path.join(DATA, "multiseed_cf.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()