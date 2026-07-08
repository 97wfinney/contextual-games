"""tests/test_analysis.py -- Phase 4 regression tests (fast subset)."""

import numpy as np
import pytest

from contextual_games.enumeration import bits_from_predicate
from contextual_games.analysis import (
    strategy_model, optimal_strategy_model, tightness_row,
    advantage_threshold, cf_threshold, noise_sweep, noisy,
)
from contextual_games.cfraction import contextual_fraction


@pytest.fixture(scope="module")
def chsh():
    bits = bits_from_predicate(2, lambda x, o: (o[0] ^ o[1]) == (x[0] & x[1]))
    val, model = optimal_strategy_model(bits, 2, restarts=6, seed=1)
    return bits, val, model


class TestStrategyModel:
    def test_value_matches_model_success(self, chsh):
        bits, val, model = chsh
        # lambda_max of the game operator == success prob of the model
        from contextual_games.enumeration import bell_scenario, game_from_bits
        from contextual_games.games import success_probability
        game = game_from_bits(bell_scenario(2), bits)
        assert success_probability(game, model) == pytest.approx(val, abs=1e-9)

    def test_model_cf_is_tsirelson(self, chsh):
        _, _, model = chsh
        assert contextual_fraction(model) == pytest.approx(
            np.sqrt(2) - 1, abs=1e-6)


class TestTightness:
    def test_chsh_slack_zero(self, chsh):
        bits, _, model = chsh
        row = tightness_row(bits, 2, model)
        assert row["slack"] == pytest.approx(0.0, abs=1e-8)
        assert row["ncf"] == pytest.approx(2 - np.sqrt(2), abs=1e-6)


class TestNoise:
    def test_chsh_thresholds_coincide(self, chsh):
        bits, _, model = chsh
        target = 1 - 1 / np.sqrt(2)
        assert advantage_threshold(bits, 2, model) == pytest.approx(
            target, abs=1e-6)
        assert cf_threshold(model) == pytest.approx(target, abs=1e-4)

    def test_cf_monotone_and_theorem4_holds(self, chsh):
        bits, _, model = chsh
        rows = noise_sweep(bits, 2, model, np.linspace(0, 0.5, 6))
        cfs = [r["cf"] for r in rows]
        assert all(a >= b - 1e-9 for a, b in zip(cfs, cfs[1:]))
        assert all(r["actual_pF"] >= r["bound_pF"] - 1e-9 for r in rows)

    def test_full_noise_is_noncontextual(self, chsh):
        _, _, model = chsh
        assert contextual_fraction(noisy(model, 1.0)) == pytest.approx(
            0.0, abs=1e-8)
