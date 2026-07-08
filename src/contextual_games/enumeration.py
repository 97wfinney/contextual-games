"""
enumeration.py -- Phase 2: enumerate small games up to symmetry.

A game on the (n,2,2) Bell scenario is one Boolean winning predicate per
context. Encoding: fix the cell order

    cell(x, o)  =  context_index(x) * 2^n + outcome_index(o)

with contexts x in itertools.product order over setting tuples and
outcomes o in itertools.product order over outcome tuples (this matches
bell_scenario() and Scenario.local_assignments). A game is then a 0/1
vector of length 4^n -- 16 bits for n=2, 64 bits for n=3 -- read as an
integer id.

Two games are equivalent if they differ by a relabelling of the
experiment: permuting parties, swapping either setting of any party, or
flipping the outcome of any (party, setting). These relabellings act as
PERMUTATIONS OF THE CELLS, so the whole symmetry group is materialised
once as an array of index permutations (group sizes: n! * 2^n * 4^n =
128 for n=2, 3072 for n=3), and the canonical form of a game is the
minimum integer over its orbit. Enumerate canonical forms, solve one
representative per class with the Phase 1 machinery, and you have the
catalogue.

Families implemented:
  * all 2^16 general two-party games (vectorised canonical sweep);
  * all XOR games for given n (win iff outcome parity equals a target
    bit per context) -- the fully-rigorous core of the catalogue;
  * uniform random general games for n = 3 (sampling the landscape).
"""

import csv
from itertools import product

import numpy as np

from .cfraction import Scenario, noncontextual_fraction
from .games import Game, classical_value, no_signalling_value


# ---------------------------------------------------------------------------
# Scenarios and encodings
# ---------------------------------------------------------------------------

def bell_scenario(n):
    """The (n,2,2) Bell scenario. Measurements are (party, setting);
    contexts are all 2^n choices of one setting per party, in
    itertools.product order over setting tuples."""
    measurements = [(p, k) for p in range(n) for k in (0, 1)]
    contexts = [
        tuple((p, x[p]) for p in range(n))
        for x in product((0, 1), repeat=n)
    ]
    return Scenario(measurements, contexts)


def game_from_bits(scenario, bits):
    """Decode a cell bit-vector into a Game on the given Bell scenario."""
    n_ctx = len(scenario.contexts)
    n_out = len(scenario.outcomes) ** len(scenario.contexts[0])
    if len(bits) != n_ctx * n_out:
        raise ValueError("bit vector does not match scenario cell count")
    preds = []
    for ci in range(n_ctx):
        wins = {
            s for oi, s in enumerate(
                product(scenario.outcomes, repeat=len(scenario.contexts[ci]))
            )
            if bits[ci * n_out + oi]
        }
        preds.append(wins)
    return Game(scenario, preds)


def bits_from_predicate(n, predicate):
    """Encode: predicate(x, o) -> bool over setting tuple x and outcome
    tuple o, both length n."""
    bits = np.zeros(4 ** n, dtype=np.uint64)
    cell = 0
    for x in product((0, 1), repeat=n):
        for o in product((0, 1), repeat=n):
            bits[cell] = 1 if predicate(x, o) else 0
            cell += 1
    return bits


def xor_game_bits(n, targets):
    """XOR game: win iff o_1 + ... + o_n = targets[x]  (mod 2)."""
    targets = tuple(targets)
    xs = list(product((0, 1), repeat=n))
    return bits_from_predicate(
        n, lambda x, o: (sum(o) % 2) == targets[xs.index(x)]
    )


# ---------------------------------------------------------------------------
# The symmetry group, as permutations of cells
# ---------------------------------------------------------------------------

def symmetry_group(n):
    """All relabellings of the (n,2,2) experiment as cell permutations.

    Generators (all involutions):
      * swap adjacent parties p, p+1;
      * flip party p's setting labels;
      * flip party p's outcome when its setting is k.
    Closed under composition by BFS. Returns array of shape (|G|, 4^n);
    a game transforms as  bits -> bits[g].
    """
    cells = [
        (x, o)
        for x in product((0, 1), repeat=n)
        for o in product((0, 1), repeat=n)
    ]
    index = {c: i for i, c in enumerate(cells)}
    N = len(cells)

    def perm(fx, fo):
        p = np.empty(N, dtype=np.int64)
        for i, (x, o) in enumerate(cells):
            p[i] = index[(fx(x), fo(x, o))]
        return p

    gens = []
    for q in range(n - 1):                      # party swaps
        def fx(x, q=q):
            x = list(x); x[q], x[q + 1] = x[q + 1], x[q]; return tuple(x)
        def fo(x, o, q=q):
            o = list(o); o[q], o[q + 1] = o[q + 1], o[q]; return tuple(o)
        gens.append(perm(fx, fo))
    for q in range(n):                          # setting flips
        def fx(x, q=q):
            x = list(x); x[q] ^= 1; return tuple(x)
        gens.append(perm(fx, lambda x, o: o))
    for q in range(n):                          # outcome flips
        for k in (0, 1):
            def fo(x, o, q=q, k=k):
                if x[q] == k:
                    o = list(o); o[q] ^= 1; return tuple(o)
                return o
            gens.append(perm(lambda x: x, fo))

    identity = np.arange(N, dtype=np.int64)
    seen = {identity.tobytes(): identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                c = g[h]
                key = c.tobytes()
                if key not in seen:
                    seen[key] = c
                    nxt.append(c)
        frontier = nxt
    return np.array(list(seen.values()))


def _powers(n_cells):
    return (1 << np.arange(n_cells - 1, -1, -1)).astype(np.uint64)


def canonical_form(bits, group):
    """Minimum integer encoding over the orbit of `bits` under `group`.
    Returns (canonical_id, canonical_bits)."""
    bits = np.asarray(bits, dtype=np.uint64)
    pw = _powers(bits.size)
    vals = (bits[group] * pw).sum(axis=1)
    i = int(np.argmin(vals))
    return int(vals[i]), bits[group[i]]


def bits_from_id(canonical_id, n_cells):
    canonical_id = int(canonical_id)   # avoid int64 overflow for 64-cell games
    return np.array(
        [(canonical_id >> s) & 1 for s in range(n_cells - 1, -1, -1)],
        dtype=np.uint64,
    )


# ---------------------------------------------------------------------------
# Enumeration of equivalence classes
# ---------------------------------------------------------------------------

def enumerate_two_party_classes():
    """Canonicalise ALL 2^16 general two-party games in one vectorised
    sweep. Returns {canonical_id: orbit_size} with orbit sizes summing
    to 65536."""
    group = symmetry_group(2)
    n_games, n_cells = 1 << 16, 16
    ids = np.arange(n_games, dtype=np.uint64)
    shifts = np.arange(n_cells - 1, -1, -1, dtype=np.uint64)
    bits = ((ids[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint64)
    pw = _powers(n_cells)
    canon = np.full(n_games, np.iinfo(np.uint64).max, dtype=np.uint64)
    for g in group:
        np.minimum(canon, bits[:, g] @ pw, out=canon)
    uniq, counts = np.unique(canon, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def enumerate_xor_classes(n):
    """Canonicalise all 2^(2^n) XOR games on n parties. Returns
    {canonical_id: {'count': games in this class within the XOR family,
                    'targets': a representative target vector}}."""
    group = symmetry_group(n)
    classes = {}
    for targets in product((0, 1), repeat=2 ** n):
        key, _ = canonical_form(xor_game_bits(n, targets), group)
        entry = classes.setdefault(key, {"count": 0, "targets": targets})
        entry["count"] += 1
    return classes


def random_game_bits(n, count, seed=0):
    """Uniform random general games (for sampling the n=3 landscape)."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield rng.integers(0, 2, size=4 ** n).astype(np.uint64)


# ---------------------------------------------------------------------------
# Solving and cataloguing
# ---------------------------------------------------------------------------

def solve_game_bits(scenario, bits):
    """All Phase 1 quantities for one game, as a flat dict (one catalogue
    row). cf_ns_strategy is the contextual fraction of AN optimal
    no-signalling strategy (the LP's choice; optima need not be unique)."""
    game = game_from_bits(scenario, bits)
    omega_c, _, k = classical_value(game)
    omega_ns, strategy = no_signalling_value(game, return_strategy=True)
    n = game.n_constraints
    return {
        "omega_c": omega_c,
        "k": k,
        "hardness": (n - k) / n,
        "omega_ns": omega_ns,
        "ns_gap": omega_ns - omega_c,
        "cf_ns_strategy": 1.0 - noncontextual_fraction(strategy),
    }


def build_catalogue(rows, path):
    """Write catalogue rows (list of dicts sharing keys) to CSV."""
    if not rows:
        raise ValueError("no rows to write")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
