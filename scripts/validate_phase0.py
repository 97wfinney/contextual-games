"""
validate.py -- Phase 0 acceptance tests.

Every check here is a value you can point to in the literature:

  1. PR box                        -> CF = 1            (strong contextuality)
  2. Bell/CHSH table (Table I)     -> CF = 1/4
  3. quantum_model reproduces Table I exactly from |Phi+> at angles (0, pi/3)
  4. Tsirelson angles (pi/8, 5pi/8)-> CF = sqrt(2) - 1
  5. Uniform model                 -> CF = 0
  6. GHZ(3) with X, Y measurements -> CF = 1            (Mermin all-vs-nothing)
  7. Dual LP returns a genuine Bell inequality whose normalised
     violation equals CF (Theorem 1)
  8. Noisy PR box: CF(eta) = max(0, 1 - 2*eta), threshold eta* = 1/2
"""

import numpy as np

from contextual_games.cfraction import (    Scenario, EmpiricalModel, uniform_model,
    contextual_fraction, bell_inequality, is_bell_inequality,
    quantum_model, equatorial_projectors, ghz_state,
)

PASS = "  [ok]"


def check(label, got, expected, atol=1e-7):
    ok = abs(got - expected) < atol
    status = PASS if ok else "  [FAIL]"
    print(f"{status} {label}: got {got:.6f}, expected {expected:.6f}")
    assert ok, label


# -- the (2,2,2) Bell scenario ----------------------------------------------

bell_scenario = Scenario(
    measurements=["a1", "a2", "b1", "b2"],
    contexts=[("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")],
)
print(bell_scenario)

# 1. PR box --------------------------------------------------------------
corr = {(0, 0): 0.5, (1, 1): 0.5, (0, 1): 0.0, (1, 0): 0.0}
anti = {(0, 1): 0.5, (1, 0): 0.5, (0, 0): 0.0, (1, 1): 0.0}
pr_box = EmpiricalModel(bell_scenario, {0: corr, 1: corr, 2: corr, 3: anti})
check("PR box CF", contextual_fraction(pr_box), 1.0)

# 2. The Bell table from the paper (Table I, left) -------------------------
mixed = {(0, 0): 3 / 8, (1, 1): 3 / 8, (0, 1): 1 / 8, (1, 0): 1 / 8}
flipd = {(0, 0): 1 / 8, (1, 1): 1 / 8, (0, 1): 3 / 8, (1, 0): 3 / 8}
bell_table = EmpiricalModel(bell_scenario, {0: corr, 1: mixed, 2: mixed, 3: flipd})
check("Bell table CF", contextual_fraction(bell_table), 0.25)

# 3. Reproduce Table I from quantum mechanics ------------------------------
phi_plus = np.zeros(4, dtype=complex)
phi_plus[0] = phi_plus[3] = 1 / np.sqrt(2)
settings = [
    [equatorial_projectors(0.0), equatorial_projectors(np.pi / 3)],  # Alice
    [equatorial_projectors(0.0), equatorial_projectors(np.pi / 3)],  # Bob
]
qm = quantum_model(phi_plus, settings)
diff = np.max(np.abs(qm.vector - bell_table.vector))
print(f"{PASS} quantum_model vs Table I: max entrywise diff = {diff:.2e}")
assert diff < 1e-9

# 4. Tsirelson model -------------------------------------------------------
settings_tsirelson = [
    [equatorial_projectors(np.pi / 8), equatorial_projectors(5 * np.pi / 8)],
    [equatorial_projectors(np.pi / 8), equatorial_projectors(5 * np.pi / 8)],
]
tsirelson = quantum_model(phi_plus, settings_tsirelson)
check("Tsirelson model CF", contextual_fraction(tsirelson), np.sqrt(2) - 1)

# 5. Uniform (non-contextual) model ----------------------------------------
check("uniform model CF", contextual_fraction(uniform_model(bell_scenario)), 0.0)

# 6. GHZ(3) with Pauli X and Y ----------------------------------------------
ghz_settings = [
    [equatorial_projectors(0.0), equatorial_projectors(np.pi / 2)]  # X, Y
    for _ in range(3)
]
ghz_model = quantum_model(ghz_state(3), ghz_settings)
print(ghz_model.scenario)
check("GHZ(3) Mermin model CF", contextual_fraction(ghz_model), 1.0)

# 7. Witnessing Bell inequality (Theorem 1) ---------------------------------
for label, model in [("Bell table", bell_table), ("Tsirelson", tsirelson)]:
    a, R, violation = bell_inequality(model)
    assert is_bell_inequality(model, a), f"{label}: dual gave a non-Bell ineq."
    check(f"{label}: normalised violation == CF",
          violation, contextual_fraction(model))

# 8. Noise sweep on the PR box ----------------------------------------------
print("\n  eta   CF(noisy PR)   expected 1-2*eta")
flat = uniform_model(bell_scenario)
for eta in [0.0, 0.1, 0.25, 0.4, 0.5, 0.75]:
    noisy = pr_box.mix(flat, 1 - eta)   # (1-eta) PR + eta uniform
    cf = contextual_fraction(noisy)
    expected = max(0.0, 1 - 2 * eta)
    flag = "ok" if abs(cf - expected) < 1e-7 else "FAIL"
    print(f"  {eta:.2f}     {cf:.6f}       {expected:.6f}   [{flag}]")
    assert abs(cf - expected) < 1e-7

print("\nAll Phase 0 validation checks passed.")
