"""tests/test_games.py -- Phase 1 regression tests."""

import numpy as np
import pytest

from contextual_games.cfraction import contextual_fraction, EmpiricalModel
from contextual_games.games import (
    chsh_game, chsh_quantum_strategy, ghz_game, ghz_quantum_strategy,
    classical_value, no_signalling_value, success_probability,
    hardness, theorem4_bound,
)

ATOL = 1e-7


class TestCHSH:
    def setup_method(self):
        self.game = chsh_game()

    def test_classical_value(self):
        wc, _, k = classical_value(self.game)
        assert wc == pytest.approx(0.75)
        assert k == 3

    def test_hardness(self):
        assert hardness(self.game) == pytest.approx(0.25)

    def test_no_signalling_value(self):
        assert no_signalling_value(self.game) == pytest.approx(1.0, abs=ATOL)

    def test_optimal_ns_strategy_is_maximally_contextual(self):
        _, strat = no_signalling_value(self.game, return_strategy=True)
        assert contextual_fraction(strat) == pytest.approx(1.0, abs=1e-6)

    def test_quantum_success_is_tsirelson(self):
        q = chsh_quantum_strategy()
        assert success_probability(self.game, q) == pytest.approx(
            (2 + np.sqrt(2)) / 4, abs=ATOL)

    def test_theorem4_tight_for_tsirelson_strategy(self):
        q = chsh_quantum_strategy()
        bound, actual = theorem4_bound(self.game, q)
        assert bound == pytest.approx((2 - np.sqrt(2)) / 4, abs=1e-6)
        assert actual == pytest.approx(bound, abs=1e-6)

    def test_theorem4_tight_for_best_classical_strategy(self):
        sc = self.game.scenario
        _, g, _ = classical_value(self.game)
        table = {
            ci: {tuple(g[sc._idx[x]] for x in C): 1.0}
            for ci, C in enumerate(sc.contexts)
        }
        det = EmpiricalModel(sc, table)
        bound, actual = theorem4_bound(self.game, det)
        assert actual == pytest.approx(bound, abs=ATOL)
        assert actual == pytest.approx(0.25, abs=ATOL)


class TestGHZ:
    def setup_method(self):
        self.game = ghz_game()

    def test_classical_value(self):
        wc, _, k = classical_value(self.game)
        assert wc == pytest.approx(0.75)
        assert k == 3

    def test_no_signalling_value(self):
        assert no_signalling_value(self.game) == pytest.approx(1.0, abs=ATOL)

    def test_pseudo_telepathy(self):
        q = ghz_quantum_strategy()
        assert success_probability(self.game, q) == pytest.approx(1.0, abs=ATOL)

    def test_strategy_is_strongly_contextual(self):
        q = ghz_quantum_strategy()
        assert contextual_fraction(q) == pytest.approx(1.0, abs=1e-6)
