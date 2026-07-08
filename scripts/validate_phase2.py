"""
validate_phase2.py -- Phase 2 acceptance tests.

Checks, in order:

  1. Symmetry group sizes match the counting argument n! * 2^n * 4^n:
     128 for n=2, 3072 for n=3.
  2. The vectorised 2-party sweep partitions all 65536 games exactly
     (orbit sizes sum to 65536).
  3. The CHSH game, encoded as bits, solves to (omega_c, omega_ns) =
     (3/4, 1) and its optimal NS strategy has CF = 1.
  4. Symmetry invariance: every game in CHSH's orbit has the same
     canonical id, and a transformed copy solved directly gives the
     same values (the physics doesn't care what we call the wires).
  5. Phase 1 games embed: padding the 4-context GHZ game with
     always-win predicates on the other 4 contexts gives
     omega_c = 7/8 and omega_ns = 1.
  6. XOR families enumerate: n=2 and n=3 class counts reported, and
     CHSH's XOR encoding (targets 0,0,0,1) lands in a class with
     omega_c = 3/4.
"""

import numpy as np

from contextual_games.enumeration import (
    bell_scenario, symmetry_group, canonical_form, bits_from_id,
    bits_from_predicate, xor_game_bits,
    enumerate_two_party_classes, enumerate_xor_classes,
    game_from_bits, solve_game_bits,
)
from contextual_games.games import classical_value, no_signalling_value

PASS = "  [ok]"


def check(label, got, expected, atol=1e-7):
    ok = abs(got - expected) < atol
    print(f"{PASS if ok else '  [FAIL]'} {label}: "
          f"got {got}, expected {expected}")
    assert ok, label


# 1. group sizes -------------------------------------------------------------
G2 = symmetry_group(2)
check("group size n=2 (2! * 4 * 16)", len(G2), 128)
G3 = symmetry_group(3)
check("group size n=3 (3! * 8 * 64)", len(G3), 3072)

# 2. the full 2-party partition ----------------------------------------------
classes2 = enumerate_two_party_classes()
total = sum(classes2.values())
check("orbit sizes sum to 2^16", total, 65536)
print(f"       number of two-party game classes: {len(classes2)}")

# 3. CHSH as a bit-vector ----------------------------------------------------
sc2 = bell_scenario(2)
chsh_bits = bits_from_predicate(
    2, lambda x, o: (o[0] ^ o[1]) == (x[0] & x[1])
)
row = solve_game_bits(sc2, chsh_bits)
check("CHSH omega_c", row["omega_c"], 0.75)
check("CHSH omega_ns", row["omega_ns"], 1.0)
check("CHSH cf of optimal NS strategy", row["cf_ns_strategy"], 1.0)

chsh_id, _ = canonical_form(chsh_bits, G2)
print(f"       CHSH canonical id: {chsh_id} "
      f"(orbit size {classes2[chsh_id]})")

# 4. symmetry invariance ------------------------------------------------------
rng = np.random.default_rng(97)
for trial in range(3):
    g = G2[rng.integers(len(G2))]
    transformed = chsh_bits[g]
    tid, _ = canonical_form(transformed, G2)
    check(f"transformed CHSH, trial {trial}: same canonical id",
          tid, chsh_id)
    trow = solve_game_bits(sc2, transformed)
    check(f"transformed CHSH, trial {trial}: same omega_c",
          trow["omega_c"], row["omega_c"])
    check(f"transformed CHSH, trial {trial}: same omega_ns",
          trow["omega_ns"], row["omega_ns"])

# 5. the GHZ game embedded in the full 8-context scenario ---------------------
mermin = {(0, 0, 0): 0, (0, 1, 1): 1, (1, 0, 1): 1, (1, 1, 0): 1}

def ghz_padded(x, o):
    if x in mermin:
        return (o[0] ^ o[1] ^ o[2]) == mermin[x]
    return True                     # always-win on the other contexts

ghz_bits = bits_from_predicate(3, ghz_padded)
sc3 = bell_scenario(3)
game3 = game_from_bits(sc3, ghz_bits)
wc, _, k = classical_value(game3)
check("padded GHZ omega_c = (3+4)/8", wc, 7 / 8)
check("padded GHZ omega_ns", no_signalling_value(game3), 1.0)

# 6. XOR families -------------------------------------------------------------
xor2 = enumerate_xor_classes(2)
counted = sum(v["count"] for v in xor2.values())
check("n=2 XOR games counted", counted, 16)
print(f"       n=2 XOR classes: {len(xor2)}")

chsh_xor_id, _ = canonical_form(xor_game_bits(2, (0, 0, 0, 1)), G2)
assert chsh_xor_id in xor2, "CHSH missing from XOR enumeration"
xrow = solve_game_bits(sc2, bits_from_id(chsh_xor_id, 16))
check("CHSH-as-XOR class omega_c", xrow["omega_c"], 0.75)

xor3 = enumerate_xor_classes(3)
counted3 = sum(v["count"] for v in xor3.values())
check("n=3 XOR games counted", counted3, 256)
print(f"       n=3 XOR classes: {len(xor3)}")

print("\nAll Phase 2 validation checks passed.")
