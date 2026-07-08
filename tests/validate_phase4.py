"""
validate_phase4.py -- Phase 4 acceptance tests.

Known values being reproduced:

  CHSH, Tsirelson-optimal strategy:
    * NCF = 2 - sqrt(2), CF = sqrt(2) - 1;
    * Theorem 4 slack = 0 (exactly tight);
    * advantage threshold eta_adv = 1 - 1/sqrt(2) ~ 0.292893
      (the textbook CHSH visibility threshold, since success is linear
      in eta and equals omega_c at visibility 1/sqrt(2));
    * CF threshold eta_cf = eta_adv: for CHSH the witnessing Bell
      inequality IS the game functional, so resource and advantage
      die together.

  Padded GHZ, Pauli X/Y strategy:
    * CF = 1 (strong contextuality), Theorem 4 bound and p_F both 0;
    * eta_adv = 1/2 exactly:  (1 - 7/8) / (1 - 3/4)  since the uniform
      strategy scores (4*1/2 + 4*1)/8 = 3/4;
    * eta_cf = 1/2, the Mermin visibility threshold.
"""

import numpy as np

from contextual_games.enumeration import bits_from_predicate
from contextual_games.analysis import (
    optimal_strategy_model, tightness_row,
    advantage_threshold, cf_threshold, noise_sweep,
)

PASS = "  [ok]"


def check(label, got, expected, atol=1e-5):
    ok = abs(got - expected) < atol
    print(f"{PASS if ok else '  [FAIL]'} {label}: "
          f"got {got:.6f}, expected {expected:.6f}")
    assert ok, label


# -- CHSH ---------------------------------------------------------------------
print("CHSH / Tsirelson strategy")
chsh = bits_from_predicate(2, lambda x, o: (o[0] ^ o[1]) == (x[0] & x[1]))
val, model = optimal_strategy_model(chsh, 2, restarts=6, seed=1)
check("strategy value", val, np.cos(np.pi / 8) ** 2)

row = tightness_row(chsh, 2, model)
check("NCF", row["ncf"], 2 - np.sqrt(2))
check("Theorem 4 bound", row["bound"], (2 - np.sqrt(2)) / 4)
check("slack (tight)", row["slack"], 0.0, atol=1e-9)

check("eta_adv = 1 - 1/sqrt(2)",
      advantage_threshold(chsh, 2, model), 1 - 1 / np.sqrt(2))
check("eta_cf  = 1 - 1/sqrt(2)",
      cf_threshold(model), 1 - 1 / np.sqrt(2), atol=1e-4)

# monotone sanity along the sweep
sweep = noise_sweep(chsh, 2, model, np.linspace(0, 0.5, 6))
cfs = [r["cf"] for r in sweep]
assert all(a >= b - 1e-9 for a, b in zip(cfs, cfs[1:])), "CF not monotone"
assert all(r["actual_pF"] >= r["bound_pF"] - 1e-9 for r in sweep), \
    "Theorem 4 violated along sweep"
print(f"{PASS} CF monotone non-increasing and Theorem 4 holds at "
      f"every eta on the sweep")

# -- padded GHZ ----------------------------------------------------------------
print("\npadded GHZ / Pauli X-Y strategy")
mermin = {(0, 0, 0): 0, (0, 1, 1): 1, (1, 0, 1): 1, (1, 1, 0): 1}
ghz = bits_from_predicate(
    3, lambda x, o: ((o[0] ^ o[1] ^ o[2]) == mermin[x]) if x in mermin
    else True)
val3, model3 = optimal_strategy_model(ghz, 3, restarts=4, seed=3)
check("strategy value (pseudo-telepathy)", val3, 1.0)

row3 = tightness_row(ghz, 3, model3)
check("CF = 1 (strong contextuality)", row3["cf"], 1.0)
check("bound = actual p_F = 0", row3["p_F"], 0.0)

check("eta_adv = 1/2 exactly",
      advantage_threshold(ghz, 3, model3), 0.5)
check("eta_cf = 1/2 (Mermin visibility)",
      cf_threshold(model3), 0.5, atol=1e-4)

print("\nAll Phase 4 validation checks passed.")
