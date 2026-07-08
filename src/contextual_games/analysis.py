"""
analysis.py -- Phase 4: the science.

Everything here composes objects that already exist. The one genuinely
new construction is `strategy_model`: it turns the optimal qubit angles
found in Phase 3 into an EmpiricalModel -- the bridge between "a quantum
strategy" (state + measurements) and "a table the contextual-fraction LP
can eat". From there:

Q2 -- TIGHTNESS OF THEOREM 4. For a strategy e on a game with hardness
(n-k)/n, Theorem 4 says  p_F >= NCF(e) * (n-k)/n.  `tightness_row`
computes both sides; the interesting quantity is the slack
p_F - bound >= 0. Slack 0 means the strategy wastes nothing: every bit
of its classical (non-contextual) fraction fails at exactly the
unavoidable rate, and its contextual fraction never fails.

Q3 -- NOISE ROBUSTNESS. Mix the strategy with white noise,
e_eta = (1-eta) e + eta u, where u is uniform on every context. Success
is exactly linear in eta, so the advantage threshold (where success
drops to omega_c) has the closed form

    eta_adv = (p_S - omega_c) / (p_S - p_U) ,

while CF(e_eta) is piecewise-linear non-increasing (mixing with a
non-contextual model is a free operation), and its zero, eta_cf, is
found by bisection on the LP. Comparing eta_adv with eta_cf per game
class asks: does the *advantage* die before, with, or after the
*resource*?

Alignment note: models built by `strategy_model` share the cell ordering
of `bell_scenario(n)` (both use itertools.product order), so their
vectors can be consumed directly by Game objects built on that scenario.
"""

import numpy as np

from .cfraction import (
    quantum_model, uniform_model,
    noncontextual_fraction, contextual_fraction,
)
from .enumeration import bell_scenario, game_from_bits
from .games import classical_value, success_probability
from .quantum import game_operator, _projectors, lower_bound_qubits


# ---------------------------------------------------------------------------
# From Phase 3 angles to a Phase 0 empirical model
# ---------------------------------------------------------------------------

def strategy_model(bits, n, angles):
    """The empirical model of the optimal qubit strategy: state = top
    eigenvector of the game operator at these angles, measurements = the
    Bloch projectors at these angles."""
    G = game_operator(bits, n, angles)
    _, vecs = np.linalg.eigh(G)
    state = vecs[:, -1]
    settings = [
        [_projectors(*angles[p, k]) for k in (0, 1)] for p in range(n)
    ]
    return quantum_model(state, settings)


def optimal_strategy_model(bits, n, restarts=10, seed=0):
    """Convenience: run the Phase 3 lower-bound search and return
    (value, model) for the best strategy found."""
    value, angles = lower_bound_qubits(bits, n, restarts=restarts, seed=seed)
    return value, strategy_model(bits, n, angles)


# ---------------------------------------------------------------------------
# Q2: tightness of Theorem 4
# ---------------------------------------------------------------------------

def tightness_row(bits, n, model):
    """Both sides of  p_F >= NCF(e) * (n-k)/n  for the strategy `model`.

    Returns a dict; the headline field is `slack` = p_F - bound >= 0
    (up to LP tolerance). slack ~ 0  <=>  the bound is tight for this
    strategy on this game.
    """
    game = game_from_bits(bell_scenario(n), bits)
    _, _, k = classical_value(game)
    nctx = game.n_constraints
    hard = (nctx - k) / nctx
    ncf = noncontextual_fraction(model)
    p_F = 1.0 - success_probability(game, model)
    bound = ncf * hard
    return {
        "k": k,
        "hardness": hard,
        "ncf": ncf,
        "cf": 1.0 - ncf,
        "p_F": p_F,
        "bound": bound,
        "slack": p_F - bound,
    }


# ---------------------------------------------------------------------------
# Q3: noise robustness
# ---------------------------------------------------------------------------

def noisy(model, eta):
    """(1 - eta) * model + eta * uniform white noise."""
    return model.mix(uniform_model(model.scenario), 1.0 - eta)


def advantage_threshold(bits, n, model):
    """Closed-form eta at which the noisy strategy's success drops to
    omega_c. Returns 0.0 if the strategy has no advantage to begin with."""
    game = game_from_bits(bell_scenario(n), bits)
    omega_c, _, _ = classical_value(game)
    p_S = success_probability(game, model)
    p_U = success_probability(game, uniform_model(model.scenario))
    if p_S <= omega_c + 1e-12:
        return 0.0
    return float((p_S - omega_c) / (p_S - p_U))


def cf_threshold(model, tol=1e-6):
    """Bisection for the eta at which CF of the noisy strategy hits 0.
    CF is non-increasing in eta (mixing with a non-contextual model is a
    free operation), so bisection is sound."""
    if contextual_fraction(model) < 1e-9:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if contextual_fraction(noisy(model, mid)) > 1e-9:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def noise_sweep(bits, n, model, etas):
    """Success, CF, and the Theorem 4 bound of the noisy strategy along a
    grid of noise rates. Returns a list of row dicts (one per eta)."""
    game = game_from_bits(bell_scenario(n), bits)
    _, _, k = classical_value(game)
    hard = (game.n_constraints - k) / game.n_constraints
    rows = []
    for eta in etas:
        m = noisy(model, eta)
        ncf = noncontextual_fraction(m)
        rows.append({
            "eta": float(eta),
            "success": success_probability(game, m),
            "cf": 1.0 - ncf,
            "bound_pF": ncf * hard,
            "actual_pF": 1.0 - success_probability(game, m),
        })
    return rows
