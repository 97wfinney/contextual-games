"""
cfraction.py -- Phase 0 reference implementation.

Implements the core objects of the Abramsky-Barbosa-Mansfield framework:

  * Scenario          -- a measurement scenario <X, M, O>
  * EmpiricalModel    -- a table of probability distributions, one per context
  * noncontextual_fraction / contextual_fraction  -- LP (3) from the paper
  * bell_inequality   -- the dual LP (4), returning the witnessing inequality
  * quantum_model     -- build an empirical model from a state + projective
                         measurements (the (n,2,2) case)

Everything is deliberately plain: numpy + scipy.optimize.linprog only.
"""

from itertools import product

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Measurement scenarios
# ---------------------------------------------------------------------------

class Scenario:
    """A measurement scenario <X, M, O>.

    Parameters
    ----------
    measurements : iterable of hashable labels (the set X)
    contexts     : iterable of tuples of measurement labels (the set M)
    n_outcomes   : size of the outcome set O (outcomes are 0..n_outcomes-1)
    """

    def __init__(self, measurements, contexts, n_outcomes=2):
        self.X = list(measurements)
        self.contexts = [tuple(C) for C in contexts]
        self.outcomes = list(range(n_outcomes))
        self._idx = {x: i for i, x in enumerate(self.X)}

        # Global assignments g : X -> O, stored as tuples ordered like self.X.
        self.global_assignments = list(
            product(self.outcomes, repeat=len(self.X))
        )

        # Local assignments <C, s>, the row index of the incidence matrix.
        self.local_assignments = [
            (ci, s)
            for ci, C in enumerate(self.contexts)
            for s in product(self.outcomes, repeat=len(C))
        ]

        self.incidence = self._build_incidence()

    def _build_incidence(self):
        """M[<C,s>, g] = 1  iff  g|_C = s."""
        m = len(self.local_assignments)
        n = len(self.global_assignments)
        M = np.zeros((m, n))
        for row, (ci, s) in enumerate(self.local_assignments):
            idxs = [self._idx[x] for x in self.contexts[ci]]
            for col, g in enumerate(self.global_assignments):
                if all(g[i] == si for i, si in zip(idxs, s)):
                    M[row, col] = 1.0
        return M

    def __repr__(self):
        return (f"Scenario(|X|={len(self.X)}, |M|={len(self.contexts)}, "
                f"|O|={len(self.outcomes)}, "
                f"globals={len(self.global_assignments)}, "
                f"locals={len(self.local_assignments)})")


# ---------------------------------------------------------------------------
# Empirical models
# ---------------------------------------------------------------------------

class EmpiricalModel:
    """A family of distributions {e_C}, one per context of a scenario.

    `table` maps context index -> {outcome tuple -> probability}.
    Internally flattened to the vector v^e aligned with
    scenario.local_assignments.
    """

    def __init__(self, scenario, table, check=True, atol=1e-9):
        self.scenario = scenario
        v = np.zeros(len(scenario.local_assignments))
        for row, (ci, s) in enumerate(scenario.local_assignments):
            v[row] = table.get(ci, {}).get(s, 0.0)
        self.vector = v
        if check:
            self._check_normalisation(atol)
            self._check_no_signalling(atol)

    # -- consistency checks -------------------------------------------------

    def context_distribution(self, ci):
        sc = self.scenario
        return {
            s: self.vector[row]
            for row, (cj, s) in enumerate(sc.local_assignments)
            if cj == ci
        }

    def _check_normalisation(self, atol):
        for ci in range(len(self.scenario.contexts)):
            total = sum(self.context_distribution(ci).values())
            if abs(total - 1.0) > atol:
                raise ValueError(
                    f"context {self.scenario.contexts[ci]} sums to {total}"
                )

    def _marginal(self, ci, sub):
        """Marginal of e_C on the sub-tuple of measurements `sub` (subset of C)."""
        C = self.scenario.contexts[ci]
        pos = [C.index(x) for x in sub]
        marg = {}
        for s, p in self.context_distribution(ci).items():
            key = tuple(s[i] for i in pos)
            marg[key] = marg.get(key, 0.0) + p
        return marg

    def _check_no_signalling(self, atol):
        ncon = len(self.scenario.contexts)
        for ci in range(ncon):
            for cj in range(ci + 1, ncon):
                overlap = tuple(
                    x for x in self.scenario.contexts[ci]
                    if x in self.scenario.contexts[cj]
                )
                if not overlap:
                    continue
                mi, mj = self._marginal(ci, overlap), self._marginal(cj, overlap)
                for key in mi:
                    if abs(mi[key] - mj.get(key, 0.0)) > atol:
                        raise ValueError(
                            f"no-signalling violated on overlap {overlap} "
                            f"of contexts {ci},{cj}: "
                            f"{mi[key]} vs {mj.get(key, 0.0)}"
                        )

    # -- convex structure ----------------------------------------------------

    def mix(self, other, lam):
        """lam * self + (1-lam) * other  (same scenario)."""
        out = EmpiricalModel.__new__(EmpiricalModel)
        out.scenario = self.scenario
        out.vector = lam * self.vector + (1 - lam) * other.vector
        return out


def uniform_model(scenario):
    """The maximally mixed (non-contextual) model: uniform in every context."""
    table = {}
    for ci, C in enumerate(scenario.contexts):
        k = len(scenario.outcomes) ** len(C)
        table[ci] = {
            s: 1.0 / k
            for s in product(scenario.outcomes, repeat=len(C))
        }
    return EmpiricalModel(scenario, table)


# ---------------------------------------------------------------------------
# The linear programs
# ---------------------------------------------------------------------------

def noncontextual_fraction(model, return_subdistribution=False):
    """LP (3): maximise 1.b  s.t.  M b <= v^e,  b >= 0.   NCF(e) = 1.b*"""
    M = model.scenario.incidence
    v = model.vector
    n = M.shape[1]
    res = linprog(
        c=-np.ones(n),            # linprog minimises
        A_ub=M, b_ub=v,
        bounds=[(0, None)] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"primal LP failed: {res.message}")
    ncf = float(-res.fun)
    return (ncf, res.x) if return_subdistribution else ncf


def contextual_fraction(model):
    return 1.0 - noncontextual_fraction(model)


def bell_inequality(model):
    """Dual LP (4) + change of variables a := |M|^{-1} 1 - y.

    Returns (a, R, normalised_violation) where a is indexed like
    scenario.local_assignments and R = 0.  By Theorem 1 the normalised
    violation equals CF(e) for contextual e.
    """
    M = model.scenario.incidence
    v = model.vector
    m, n = M.shape
    res = linprog(
        c=v,
        A_ub=-M.T, b_ub=-np.ones(n),   # M^T y >= 1
        bounds=[(0, None)] * m,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"dual LP failed: {res.message}")
    y = res.x
    n_contexts = len(model.scenario.contexts)
    a = (1.0 / n_contexts) - y

    # algebraic bound ||a|| = sum over contexts of the max coefficient
    norm = 0.0
    for ci in range(n_contexts):
        rows = [r for r, (cj, _) in enumerate(model.scenario.local_assignments)
                if cj == ci]
        norm += max(a[r] for r in rows)

    value = float(a @ v)
    violation = max(0.0, value) / norm if norm > 0 else 0.0
    return a, 0.0, violation


def is_bell_inequality(model, a, atol=1e-9):
    """Check  M^T a <= 0 : satisfied (with bound 0) by every NC model."""
    return bool(np.all(model.scenario.incidence.T @ a <= atol))


# ---------------------------------------------------------------------------
# Quantum models on (n, k, 2) scenarios
# ---------------------------------------------------------------------------

def equatorial_projectors(phi):
    """Projectors for the +/-1 eigenvectors of  cos(phi) X + sin(phi) Y.

    Outcome 0 <-> +1 eigenvalue, outcome 1 <-> -1 eigenvalue.
    """
    plus = np.array([1.0, np.exp(1j * phi)]) / np.sqrt(2)
    minus = np.array([1.0, -np.exp(1j * phi)]) / np.sqrt(2)
    return [np.outer(w, w.conj()) for w in (plus, minus)]


def quantum_model(state, settings_per_party):
    """Empirical model from local projective measurements on a pure state.

    settings_per_party[p][k] = [P_outcome0, P_outcome1]   (2x2 projectors)
    Measurement labels are pairs (party, setting); contexts are all choices
    of one setting per party.
    """
    n_parties = len(settings_per_party)
    measurements = [
        (p, k)
        for p in range(n_parties)
        for k in range(len(settings_per_party[p]))
    ]
    choices = product(*[range(len(s)) for s in settings_per_party])
    contexts = [
        tuple((p, choice[p]) for p in range(n_parties)) for choice in choices
    ]
    scenario = Scenario(measurements, contexts, n_outcomes=2)

    table = {}
    for ci, C in enumerate(scenario.contexts):
        dist = {}
        for s in product([0, 1], repeat=n_parties):
            P = np.eye(1)
            for (p, k), o in zip(C, s):
                P = np.kron(P, settings_per_party[p][k][o])
            prob = np.real(state.conj() @ (P @ state))
            dist[s] = max(float(prob), 0.0)
        table[ci] = dist
    return EmpiricalModel(scenario, table)


def ghz_state(n):
    """(|0...0> + |1...1>)/sqrt(2) as a flat vector of length 2^n."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = psi[-1] = 1.0 / np.sqrt(2)
    return psi
