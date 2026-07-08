"""
quantum.py -- Phase 3: quantum values of binary-outcome games.

Two complementary computations, giving a bracket  lower <= omega_q <= upper:

LOWER BOUND (explicit strategies): restrict to one qubit per party with
projective measurements. For fixed measurement angles the optimal shared
state is the top eigenvector of the game operator

    G = (1/|M|) sum_x sum_{o in W_x}  (x)_p P^{o_p}_{p, x_p}

so the bound is max over angles of lambda_max(G), optimised by
multi-start local search (scipy). Any value found is achievable, hence a
genuine lower bound on omega_q (qubit strategies are a subset of all
quantum strategies).

UPPER BOUND (moment-matrix relaxation): the NPA-style intermediate level
whose monomials are products of AT MOST ONE +/-1 OBSERVABLE PER PARTY
(for two parties this is the level often written Q^{1+AB}). Because
outcomes are binary, every probability is a linear combination of subset
correlators,

    P(o|x) = 2^{-n} sum_{S subseteq parties} (-1)^{sum_{p in S} o_p}
                     < prod_{p in S} O_{p, x_p} > ,

and all of those correlators appear in the moment matrix, so this one
relaxation handles ARBITRARY binary games, not just XOR games. Moment
variables are real (WLOG: conjugating a strategy preserves the game value
and averaging the two moment matrices gives a real PSD one). Explicit
positivity constraints P(o|x) >= 0 are added, which also guarantees
upper <= omega_ns.

Exactness: for BIPARTITE XOR games this level is exact (Tsirelson), so
lower and upper meet and omega_q is certified. In general it is an upper
bound that may be loose; when the bracket does not close, report the
interval -- do not report a point value.
"""

from itertools import product

import numpy as np
from scipy.optimize import minimize

try:
    import cvxpy as cp
except ImportError as exc:                                  # pragma: no cover
    raise ImportError("Phase 3 needs cvxpy:  pip install cvxpy") from exc


# ---------------------------------------------------------------------------
# Lower bounds: explicit qubit strategies
# ---------------------------------------------------------------------------

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _projectors(theta, phi):
    """(P_+, P_-) for the qubit observable  n.sigma  with Bloch angles
    (theta, phi). Outcome 0 <-> +1 eigenvalue."""
    n_sigma = (np.sin(theta) * np.cos(phi) * _SX
               + np.sin(theta) * np.sin(phi) * _SY
               + np.cos(theta) * _SZ)
    return (_I2 + n_sigma) / 2, (_I2 - n_sigma) / 2


def game_operator(bits, n, angles):
    """The game operator on (C^2)^{tensor n} for measurement angles of
    shape (n, 2, 2): angles[party, setting] = (theta, phi)."""
    n_ctx = 2 ** n
    n_out = 2 ** n
    projs = [
        [_projectors(*angles[p, k]) for k in (0, 1)] for p in range(n)
    ]
    G = np.zeros((2 ** n, 2 ** n), dtype=complex)
    cell = 0
    for x in product((0, 1), repeat=n):
        for o in product((0, 1), repeat=n):
            if bits[cell]:
                M = np.eye(1, dtype=complex)
                for p in range(n):
                    M = np.kron(M, projs[p][x[p]][o[p]])
                G += M
            cell += 1
    return G / n_ctx


def lower_bound_qubits(bits, n, restarts=12, seed=0):
    """max over angles of lambda_max(G): the best qubit strategy found.
    Returns (value, angles). A genuine lower bound on omega_q.

    The first start is seeded with equatorial Pauli X/Y angles (optimal
    for GHZ/Mermin-type games); the rest are random."""
    rng = np.random.default_rng(seed)

    def objective(flat):
        angles = flat.reshape(n, 2, 2)
        G = game_operator(bits, n, angles)
        return -np.linalg.eigvalsh(G)[-1]

    xy_seed = np.array(
        [[[np.pi / 2, 0.0], [np.pi / 2, np.pi / 2]]] * n
    ).ravel()
    starts = [xy_seed] + [
        np.stack([
            rng.uniform(0, np.pi, size=(n, 2)),        # theta
            rng.uniform(0, 2 * np.pi, size=(n, 2)),    # phi
        ], axis=-1).ravel()
        for _ in range(max(0, restarts - 1))
    ]

    best_val, best_angles = -np.inf, None
    for x0 in starts:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 1200, "xatol": 1e-7,
                                "fatol": 1e-10})
        if -res.fun > best_val:
            best_val, best_angles = -res.fun, res.x.reshape(n, 2, 2)
    return float(best_val), best_angles


def strategy_value(bits, n, angles):
    """Value of the qubit strategy with given angles (state = top
    eigenvector of G). Useful for evaluating known strategies exactly."""
    G = game_operator(bits, n, angles)
    return float(np.linalg.eigvalsh(G)[-1])


# ---------------------------------------------------------------------------
# Upper bounds: moment-matrix relaxation
# ---------------------------------------------------------------------------

def _monomials(n):
    """All products of at most one observable per party: a monomial is a
    dict {party: setting}. Ordered with the identity (empty dict) first."""
    monos = []
    for pattern in product((None, 0, 1), repeat=n):
        monos.append({p: s for p, s in enumerate(pattern) if s is not None})
    monos.sort(key=len)
    return monos


def _entry_key(u, v, n):
    """Canonical key of the moment  < u^dagger v >.

    Per party the reduced word is: () if absent from both or if the same
    setting cancels (O^2 = 1); (s,) if present in exactly one; (s, t) with
    s != t if present in both with different settings. Different parties
    commute, so the key is the tuple of per-party words; hermitian
    conjugation reverses each word, and the canonical key is the lex-min
    of the key and its conjugate (real moment matrix identifies both).
    """
    key, conj = [], []
    for p in range(n):
        a, b = u.get(p), v.get(p)
        if a is None and b is None:
            w = ()
        elif a is None:
            w = (b,)
        elif b is None:
            w = (a,)
        elif a == b:
            w = ()
        else:
            w = (a, b)
        key.append(w)
        conj.append(tuple(reversed(w)))
    key, conj = tuple(key), tuple(conj)
    return min(key, conj)


def _correlator_key(x, S, n):
    """Key of the subset correlator  < prod_{p in S} O_{p, x_p} >."""
    return tuple((x[p],) if p in S else () for p in range(n))


def npa_upper_bound(bits, n, solver="CLARABEL", verbose=False):
    """SDP upper bound on omega_q at the <=1-observable-per-party level,
    with explicit probability-positivity constraints."""
    monos = _monomials(n)
    dim = len(monos)

    # group moment-matrix entries by canonical key
    keys = {}
    positions = {}
    for i, u in enumerate(monos):
        for j, v in enumerate(monos):
            k = _entry_key(u, v, n)
            if k not in keys:
                keys[k] = len(keys)
            positions.setdefault(k, []).append((i, j))
    identity_key = tuple(() for _ in range(n))

    y = cp.Variable(len(keys))

    indicator = {}
    for k, pos in positions.items():
        A = np.zeros((dim, dim))
        for (i, j) in pos:
            A[i, j] = 1.0
        indicator[k] = A
    M = sum(y[keys[k]] * A for k, A in indicator.items())

    # probabilities as linear expressions in the moment variables
    n_ctx = 2 ** n
    parties = list(range(n))
    subsets = [tuple(S) for r in range(n + 1)
               for S in __import__("itertools").combinations(parties, r)]

    def prob_expr(x, o):
        expr = 0
        for S in subsets:
            sign = (-1) ** sum(o[p] for p in S)
            k = _correlator_key(x, set(S), n)
            expr = expr + sign * y[keys[k]]
        return expr / (2 ** n)

    constraints = [M >> 0, y[keys[identity_key]] == 1]
    objective = 0
    cell = 0
    for x in product((0, 1), repeat=n):
        for o in product((0, 1), repeat=n):
            p_expr = prob_expr(x, o)
            constraints.append(p_expr >= 0)
            if bits[cell]:
                objective = objective + p_expr
            cell += 1
    objective = objective / n_ctx

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver, verbose=verbose)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP failed: status {problem.status}")
    return float(problem.value)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

def quantum_bracket(bits, n, restarts=20, seed=0, tol=1e-4):
    """Bracket [lower, upper] for omega_q; certified iff they meet.

    Returns dict with lower, upper, certified, gap. Report the interval
    when certified is False -- a bare upper bound is not omega_q.
    """
    lower, _ = lower_bound_qubits(bits, n, restarts=restarts, seed=seed)
    upper = npa_upper_bound(bits, n)
    # numerical guards: lower is achievable, upper is a relaxation
    gap = max(0.0, upper - lower)
    return {
        "omega_q_lower": lower,
        "omega_q_upper": upper,
        "gap": gap,
        "certified": gap < tol,
    }
