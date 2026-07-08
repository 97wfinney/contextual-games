"""
validate_phase1.py -- Phase 1 acceptance tests.

Known values being reproduced:

  CHSH game:  omega_c = 3/4 (k = 3, hardness 1/4),  omega_ns = 1.
              Optimal NS strategy is a PR box  -> CF = 1.
              Tsirelson strategy: p_S = (2+sqrt2)/4 ~ 0.8536, and the
              Theorem 4 bound  p_F >= NCF * (n-k)/n  is EXACTLY TIGHT:
              (2-sqrt2) * 1/4 = (2-sqrt2)/4 = actual failure rate.
  GHZ game:   omega_c = 3/4,  omega_ns = 1, and the quantum X/Y strategy
              on |GHZ(3)> wins with certainty (pseudo-telepathy), which by
              Theorem 4 forces its strategy model to have NCF = 0, i.e.
              CF = 1 (strong contextuality).
"""

import numpy as np

from contextual_games.cfraction import contextual_fraction
from contextual_games.games import (
    chsh_game, chsh_quantum_strategy, ghz_game, ghz_quantum_strategy,
    classical_value, no_signalling_value, success_probability,
    hardness, theorem4_bound,
)

PASS = "  [ok]"


def check(label, got, expected, atol=1e-7):
    ok = abs(got - expected) < atol
    print(f"{PASS if ok else '  [FAIL]'} {label}: "
          f"got {got:.6f}, expected {expected:.6f}")
    assert ok, label


# -- CHSH --------------------------------------------------------------------
print("CHSH game")
chsh = chsh_game()

wc, g_best, k = classical_value(chsh)
check("omega_c", wc, 0.75)
check("k (max jointly satisfiable)", k, 3)
check("hardness (n-k)/n", hardness(chsh), 0.25)
print(f"       best deterministic assignment (a1,a2,b1,b2) = {g_best}")

wns, ns_strategy = no_signalling_value(chsh, return_strategy=True)
check("omega_ns", wns, 1.0)
check("CF of optimal NS strategy (PR box)",
      contextual_fraction(ns_strategy), 1.0)

q = chsh_quantum_strategy()
pS = success_probability(chsh, q)
check("quantum strategy p_S", pS, (2 + np.sqrt(2)) / 4)

bound, actual = theorem4_bound(chsh, q)
check("Theorem 4 bound  NCF*(n-k)/n", bound, (2 - np.sqrt(2)) / 4)
check("actual p_F", actual, (2 - np.sqrt(2)) / 4)
check("tightness gap (actual - bound)", actual - bound, 0.0)

# classical strategies also sit exactly on the bound: NCF = 1, p_F = 1/4
from contextual_games.cfraction import EmpiricalModel
det_table = {}
sc = chsh.scenario
for ci, C in enumerate(sc.contexts):
    s = tuple(g_best[sc._idx[x]] for x in C)
    det_table[ci] = {s: 1.0}
det = EmpiricalModel(sc, det_table)
bound_d, actual_d = theorem4_bound(chsh, det)
check("Theorem 4 tight for best classical strategy too",
      actual_d - bound_d, 0.0)

# -- GHZ ---------------------------------------------------------------------
print("\nGHZ game")
ghz = ghz_game()

wc, _, k = classical_value(ghz)
check("omega_c", wc, 0.75)
check("hardness", hardness(ghz), 0.25)

wns = no_signalling_value(ghz)
check("omega_ns", wns, 1.0)

q3 = ghz_quantum_strategy()
check("quantum strategy p_S (pseudo-telepathy)",
      success_probability(ghz, q3), 1.0)
check("CF of the GHZ strategy (strong contextuality forced by Thm 4)",
      contextual_fraction(q3), 1.0)

bound, actual = theorem4_bound(ghz, q3)
check("Theorem 4 consistent: bound", bound, 0.0)
check("Theorem 4 consistent: actual p_F", actual, 0.0)

print("\nAll Phase 1 validation checks passed.")
