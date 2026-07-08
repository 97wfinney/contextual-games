"""tests/test_quantum.py -- Phase 3 regression tests.

Kept fast (~30s): CHSH certification, sandwich property, and the padded
GHZ pseudo-telepathy. The full catalogue classification lives in
scripts/compute_quantum_values.py, not in the test suite.
"""

import numpy as np
import pytest

from contextual_games.enumeration import (
    bell_scenario, bits_from_predicate, xor_game_bits, game_from_bits,
)
from contextual_games.games import classical_value, no_signalling_value
from contextual_games.quantum import (
    quantum_bracket, strategy_value, npa_upper_bound, lower_bound_qubits,
)

TSIRELSON = np.cos(np.pi / 8) ** 2


@pytest.fixture(scope="module")
def chsh_bits():
    return bits_from_predicate(2, lambda x, o: (o[0] ^ o[1]) == (x[0] & x[1]))


class TestCHSH:
    def test_upper_bound_is_tsirelson(self, chsh_bits):
        assert npa_upper_bound(chsh_bits, 2) == pytest.approx(
            TSIRELSON, abs=1e-5)

    def test_lower_bound_reaches_tsirelson(self, chsh_bits):
        lo, _ = lower_bound_qubits(chsh_bits, 2, restarts=6, seed=1)
        assert lo == pytest.approx(TSIRELSON, abs=1e-6)

    def test_bracket_certifies(self, chsh_bits):
        br = quantum_bracket(chsh_bits, 2, restarts=6, seed=1)
        assert br["certified"]

    def test_sandwich(self, chsh_bits):
        game = game_from_bits(bell_scenario(2), chsh_bits)
        wc, _, _ = classical_value(game)
        wns = no_signalling_value(game)
        br = quantum_bracket(chsh_bits, 2, restarts=6, seed=1)
        assert wc - 1e-6 <= br["omega_q_lower"]
        assert br["omega_q_upper"] <= wns + 1e-6


class TestPseudoTelepathy:
    @pytest.fixture(scope="class")
    def ghz_bits(self):
        mermin = {(0, 0, 0): 0, (0, 1, 1): 1, (1, 0, 1): 1, (1, 1, 0): 1}
        return bits_from_predicate(
            3,
            lambda x, o: ((o[0] ^ o[1] ^ o[2]) == mermin[x])
            if x in mermin else True,
        )

    def test_explicit_xy_strategy_wins(self, ghz_bits):
        xy = np.array([[[np.pi / 2, 0.0], [np.pi / 2, np.pi / 2]]] * 3)
        assert strategy_value(ghz_bits, 3, xy) == pytest.approx(1.0)

    def test_certified_pseudo_telepathy(self, ghz_bits):
        br = quantum_bracket(ghz_bits, 3, restarts=3, seed=3)
        assert br["certified"]
        assert br["omega_q_lower"] == pytest.approx(1.0, abs=1e-4)
        wc, _, _ = classical_value(game_from_bits(bell_scenario(3), ghz_bits))
        assert wc == pytest.approx(7 / 8)


class TestTrivialAndClassical:
    def test_trivial_game(self):
        br = quantum_bracket(np.ones(16, dtype=np.uint64), 2,
                             restarts=2, seed=0)
        assert br["certified"]
        assert br["omega_q_upper"] == pytest.approx(1.0, abs=1e-6)

    def test_classical_xor_game(self):
        br = quantum_bracket(xor_game_bits(2, (0, 0, 0, 0)), 2,
                             restarts=3, seed=2)
        assert br["certified"]
        assert br["omega_q_lower"] == pytest.approx(1.0, abs=1e-6)
