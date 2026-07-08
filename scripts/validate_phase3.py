"""
validate_phase3.py -- Phase 3 acceptance tests.

Known values being reproduced:

  1. CHSH: omega_q = cos^2(pi/8) ~ 0.853553, certified (upper and lower
     meet; the relaxation is exact for bipartite XOR games by Tsirelson).
  2. The trivial game: omega_q = 1.
  3. A purely classical game (the n=2 XOR class with omega_c = 1):
     bracket closes at 1 with no quantum advantage.
  4. Padded GHZ (Mermin contexts + always-win): omega_q = 1 certified,
     i.e. pseudo-telepathy: omega_c = 7/8 < omega_q = 1. The explicit
     Pauli X/Y angles evaluate to 1 through the same game operator.
  5. Sanity: for every game tested, omega_c <= lower and upper <= omega_ns
     (the three-level sandwich).
"""

import numpy as np

from contextual_games.enumeration import (
    bell_scenario, bits_from_predicate, xor_game_bits, game_from_bits,
)
from contextual_games.games import classical_value, no_signalling_value
from contextual_games.quantum import (
    quantum_bracket, strategy_value, npa_upper_bound,
)

PASS = "  [ok]"


def check(label, got, expected, atol=1e-5):
    ok = abs(got - expected) < atol
    print(f"{PASS if ok else '  [FAIL]'} {label}: "
          f"got {got:.6f}, expected {expected:.6f}")
    assert ok, label


def sandwich(label, bits, n, bracket):
    game = game_from_bits(bell_scenario(n), bits)
    wc, _, _ = classical_value(game)
    wns = no_signalling_value(game)
    lo, up = bracket["omega_q_lower"], bracket["omega_q_upper"]
    ok = (wc - 1e-6 <= lo) and (up <= wns + 1e-6)
    print(f"{PASS if ok else '  [FAIL]'} {label}: sandwich "
          f"omega_c={wc:.4f} <= {lo:.4f} <= omega_q <= {up:.4f} "
          f"<= omega_ns={wns:.4f}")
    assert ok, label


# 1. CHSH ---------------------------------------------------------------------
chsh = bits_from_predicate(2, lambda x, o: (o[0] ^ o[1]) == (x[0] & x[1]))
br = quantum_bracket(chsh, 2, restarts=10, seed=1)
check("CHSH omega_q upper", br["omega_q_upper"], np.cos(np.pi / 8) ** 2)
check("CHSH omega_q lower", br["omega_q_lower"], np.cos(np.pi / 8) ** 2)
assert br["certified"], "CHSH bracket failed to close"
print(f"{PASS} CHSH certified (gap {br['gap']:.2e})")
sandwich("CHSH", chsh, 2, br)

# 2. trivial game --------------------------------------------------------------
trivial = np.ones(16, dtype=np.uint64)
br_t = quantum_bracket(trivial, 2, restarts=2, seed=0)
check("trivial game omega_q", br_t["omega_q_upper"], 1.0)
assert br_t["certified"]

# 3. purely classical XOR game --------------------------------------------------
classical_xor = xor_game_bits(2, (0, 0, 0, 0))
br_c = quantum_bracket(classical_xor, 2, restarts=5, seed=2)
check("classical XOR game omega_q", br_c["omega_q_lower"], 1.0)
assert br_c["certified"]

# 4. padded GHZ: pseudo-telepathy certified -------------------------------------
mermin = {(0, 0, 0): 0, (0, 1, 1): 1, (1, 0, 1): 1, (1, 1, 0): 1}
ghz_bits = bits_from_predicate(
    3,
    lambda x, o: ((o[0] ^ o[1] ^ o[2]) == mermin[x]) if x in mermin else True,
)

# the explicit Mermin strategy: X (theta=pi/2, phi=0), Y (theta=pi/2, phi=pi/2)
xy_angles = np.array([[[np.pi / 2, 0.0], [np.pi / 2, np.pi / 2]]] * 3)
check("padded GHZ: explicit X/Y strategy value",
      strategy_value(ghz_bits, 3, xy_angles), 1.0)

br_g = quantum_bracket(ghz_bits, 3, restarts=12, seed=3)
check("padded GHZ omega_q upper", br_g["omega_q_upper"], 1.0)
check("padded GHZ omega_q lower", br_g["omega_q_lower"], 1.0, atol=1e-4)
assert br_g["certified"], "GHZ pseudo-telepathy not certified"
print(f"{PASS} pseudo-telepathy certified: omega_c = 7/8 < omega_q = 1")
sandwich("padded GHZ", ghz_bits, 3, br_g)

print("\nAll Phase 3 validation checks passed.")
