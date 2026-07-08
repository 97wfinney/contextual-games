"""
games.py -- Phase 1: non-local games, classical and no-signalling values.

A game lives on a measurement scenario <X, M, O>: the referee picks a
context C uniformly at random, the players (jointly) return an outcome
tuple s in O^C, and they win iff s satisfies the winning predicate for C.
A strategy is exactly an empirical model on the scenario (a probability
table satisfying no-signalling), so the three values are optimisations of
the same objective over three nested sets:

    omega_c    -- deterministic global assignments (columns of the
                  incidence matrix), by exhaustive search;
    omega_q    -- quantum strategies (Phase 3, via NPA);
    omega_ns   -- the whole no-signalling polytope, by LP.

Also implemented: game hardness (n-k)/n via k-consistency, and the
Theorem 4 bound  p_F >= NCF(e) * (n-k)/n.

NOTE: adjust the relative import below to whatever you named your Phase 0
module inside src/contextual_games/.
"""

from itertools import product

import numpy as np
from scipy.optimize import linprog

from .cfraction import (
    Scenario, EmpiricalModel,
    noncontextual_fraction, quantum_model, equatorial_projectors, ghz_state,
)


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class Game:
    """A non-local game on a measurement scenario.

    Parameters
    ----------
    scenario   : Scenario
    predicates : list, one entry per context (in scenario.contexts order).
        Each entry is either a callable  outcome_tuple -> bool,  or an
        explicit collection of winning outcome tuples.
    """

    def __init__(self, scenario, predicates):
        if len(predicates) != len(scenario.contexts):
            raise ValueError("need exactly one predicate per context")
        self.scenario = scenario
        self.winning = []
        for ci, C in enumerate(scenario.contexts):
            pred = predicates[ci]
            if callable(pred):
                wins = frozenset(
                    s for s in product(scenario.outcomes, repeat=len(C))
                    if pred(s)
                )
            else:
                wins = frozenset(map(tuple, pred))
            self.winning.append(wins)

    @property
    def n_constraints(self):
        return len(self.scenario.contexts)

    def wins(self, ci, s):
        return s in self.winning[ci]


# ---------------------------------------------------------------------------
# The three-level values (levels 1 and 3 in this phase)
# ---------------------------------------------------------------------------

def classical_value(game):
    """omega_c by exhaustive search over deterministic global assignments.

    Returns (omega_c, best_assignment, k) where k is the maximum number of
    contexts jointly winnable by a single assignment (the k of
    'k-consistency' in Theorem 4).
    """
    sc = game.scenario
    best_k, best_g = -1, None
    for g in sc.global_assignments:
        k = 0
        for ci, C in enumerate(sc.contexts):
            s = tuple(g[sc._idx[x]] for x in C)
            if game.wins(ci, s):
                k += 1
        if k > best_k:
            best_k, best_g = k, g
    n = game.n_constraints
    return best_k / n, best_g, best_k


def hardness(game):
    """(n - k)/n : the fraction of constraints that must fail classically."""
    _, _, k = classical_value(game)
    n = game.n_constraints
    return (n - k) / n


def success_probability(game, model):
    """Average winning probability of a strategy (empirical model),
    with the context chosen uniformly at random."""
    sc = game.scenario
    total = 0.0
    for row, (ci, s) in enumerate(sc.local_assignments):
        if game.wins(ci, s):
            total += model.vector[row]
    return total / game.n_constraints


def no_signalling_value(game, return_strategy=False):
    """omega_ns by LP over the no-signalling polytope.

    Unlike the contextual-fraction LP (whose variables range over global
    assignments), the variables here are the table entries themselves:
    one variable p[<C,s>] per cell. Constraints:

      * normalisation:   each context's distribution sums to 1;
      * no-signalling:   for every pair of overlapping contexts and every
                         outcome pattern on the overlap, the two marginals
                         agree;
      * non-negativity.

    Objective: the average winning mass, (1/n) sum over winning cells.
    """
    sc = game.scenario
    m = len(sc.local_assignments)
    ncon = game.n_constraints

    # objective vector
    w = np.zeros(m)
    for row, (ci, s) in enumerate(sc.local_assignments):
        if game.wins(ci, s):
            w[row] = 1.0 / ncon

    A_eq, b_eq = [], []

    # normalisation, one row per context
    for ci in range(ncon):
        row = np.zeros(m)
        for r, (cj, _) in enumerate(sc.local_assignments):
            if cj == ci:
                row[r] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    # no-signalling, one row per (context pair, overlap pattern)
    for ci in range(ncon):
        for cj in range(ci + 1, ncon):
            overlap = tuple(x for x in sc.contexts[ci]
                            if x in sc.contexts[cj])
            if not overlap:
                continue
            pos_i = [sc.contexts[ci].index(x) for x in overlap]
            pos_j = [sc.contexts[cj].index(x) for x in overlap]
            for t in product(sc.outcomes, repeat=len(overlap)):
                row = np.zeros(m)
                for r, (ck, s) in enumerate(sc.local_assignments):
                    if ck == ci and tuple(s[p] for p in pos_i) == t:
                        row[r] += 1.0
                    elif ck == cj and tuple(s[p] for p in pos_j) == t:
                        row[r] -= 1.0
                A_eq.append(row)
                b_eq.append(0.0)

    res = linprog(
        c=-w,                      # linprog minimises
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=[(0.0, 1.0)] * m,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"no-signalling LP failed: {res.message}")
    omega_ns = float(-res.fun)

    if not return_strategy:
        return omega_ns

    table = {}
    for row, (ci, s) in enumerate(sc.local_assignments):
        table.setdefault(ci, {})[s] = float(res.x[row])
    strategy = EmpiricalModel(sc, table, check=True, atol=1e-6)
    return omega_ns, strategy


def theorem4_bound(game, model):
    """The lower bound of Theorem 4:  p_F >= NCF(e) * (n-k)/n.

    Returns (bound, actual_failure_probability). The claim being tested in
    Phase 4 is when these are equal.
    """
    bound = noncontextual_fraction(model) * hardness(game)
    actual = 1.0 - success_probability(game, model)
    return bound, actual


# ---------------------------------------------------------------------------
# Restriction of models to a sub-family of contexts
# ---------------------------------------------------------------------------

def restrict_model(model, contexts):
    """A model on the sub-scenario keeping only the given contexts.

    Contexts must appear verbatim (same measurement order) in the original
    scenario. This is 'translation of measurements', a free operation, so
    CF can only decrease.
    """
    old = model.scenario
    sub = Scenario(old.X, contexts, n_outcomes=len(old.outcomes))
    table = {}
    for new_ci, C in enumerate(sub.contexts):
        old_ci = old.contexts.index(C)
        table[new_ci] = model.context_distribution(old_ci)
    return EmpiricalModel(sub, table, check=True, atol=1e-9)


# ---------------------------------------------------------------------------
# The two canonical games, with their optimal quantum strategies
# ---------------------------------------------------------------------------

def chsh_game():
    """CHSH: settings x,y in {0,1}, win iff  o_A xor o_B = x AND y.
    Context order matches the Phase 0 Bell scenario:
    (a1,b1), (a1,b2), (a2,b1), (a2,b2)  with a1 <-> x=0, a2 <-> x=1."""
    sc = Scenario(
        ["a1", "a2", "b1", "b2"],
        [("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")],
    )
    preds = [
        (lambda s, x=x, y=y: (s[0] ^ s[1]) == (x & y))
        for (x, y) in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]
    return Game(sc, preds)


def chsh_quantum_strategy():
    """The Tsirelson-optimal strategy as an empirical model on the CHSH
    scenario: |Phi+> with equatorial angles chosen so that
    E(a_i, b_j) = cos(phi_i + phi_j) has the (+,+,+,-)/sqrt2 pattern."""
    phi_plus = np.zeros(4, dtype=complex)
    phi_plus[0] = phi_plus[3] = 1 / np.sqrt(2)
    settings = [
        [equatorial_projectors(0.0), equatorial_projectors(-np.pi / 2)],   # Alice
        [equatorial_projectors(np.pi / 4), equatorial_projectors(-np.pi / 4)],  # Bob
    ]
    qm = quantum_model(phi_plus, settings)
    # relabel that scenario's (party,setting) measurements onto a1..b2 by
    # rebuilding the table on the Phase 0 Bell scenario (same context order).
    sc = chsh_game().scenario
    table = {ci: qm.context_distribution(ci) for ci in range(4)}
    return EmpiricalModel(sc, table, check=True, atol=1e-9)


def ghz_game():
    """The GHZ/Mermin game. Measurements (party, setting), setting 0 = X,
    setting 1 = Y. Referee's questions are the four even-parity setting
    triples; win iff the outcome parity matches the Mermin constraint:

        XXX : o1+o2+o3 = 0 (mod 2)
        XYY, YXY, YYX : o1+o2+o3 = 1 (mod 2)
    """
    measurements = [(p, k) for p in range(3) for k in (0, 1)]
    contexts = [
        ((0, 0), (1, 0), (2, 0)),   # XXX
        ((0, 0), (1, 1), (2, 1)),   # XYY
        ((0, 1), (1, 0), (2, 1)),   # YXY
        ((0, 1), (1, 1), (2, 0)),   # YYX
    ]
    parities = [0, 1, 1, 1]
    preds = [
        (lambda s, par=par: (s[0] ^ s[1] ^ s[2]) == par)
        for par in parities
    ]
    return Game(Scenario(measurements, contexts), preds)


def ghz_quantum_strategy():
    """Pauli X/Y measurements on the tripartite GHZ state, restricted to
    the four game contexts. Wins the GHZ game with certainty."""
    settings = [
        [equatorial_projectors(0.0), equatorial_projectors(np.pi / 2)]
        for _ in range(3)
    ]
    full = quantum_model(ghz_state(3), settings)
    return restrict_model(full, ghz_game().scenario.contexts)
