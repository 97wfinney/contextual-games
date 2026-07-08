"""tests/test_cfraction.py -- Phase 0 regression tests.

Ports the acceptance checks from scripts/validate_phase0.py: every value here
is one you can point to in the literature (Abramsky-Barbosa-Mansfield,
arXiv:1705.07918).
"""

import numpy as np
import pytest

from contextual_games.cfraction import (
    Scenario, EmpiricalModel, uniform_model,
    contextual_fraction, bell_inequality, is_bell_inequality,
    quantum_model, equatorial_projectors, ghz_state,
)

ATOL = 1e-7


def bell_scenario():
    """The (2, 2, 2) Bell scenario used throughout Table I of the paper."""
    return Scenario(
        measurements=["a1", "a2", "b1", "b2"],
        contexts=[("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")],
    )


# correlated / anti-correlated single-context distributions
CORR = {(0, 0): 0.5, (1, 1): 0.5, (0, 1): 0.0, (1, 0): 0.0}
ANTI = {(0, 1): 0.5, (1, 0): 0.5, (0, 0): 0.0, (1, 1): 0.0}


def bell_table_model(scenario):
    """Table I (left) of the paper: CF = 1/4."""
    mixed = {(0, 0): 3 / 8, (1, 1): 3 / 8, (0, 1): 1 / 8, (1, 0): 1 / 8}
    flipd = {(0, 0): 1 / 8, (1, 1): 1 / 8, (0, 1): 3 / 8, (1, 0): 3 / 8}
    return EmpiricalModel(scenario, {0: CORR, 1: mixed, 2: mixed, 3: flipd})


def tsirelson_model(scenario):
    """|Phi+> measured at Tsirelson angles (pi/8, 5pi/8) on both parties."""
    phi_plus = np.zeros(4, dtype=complex)
    phi_plus[0] = phi_plus[3] = 1 / np.sqrt(2)
    settings = [
        [equatorial_projectors(np.pi / 8), equatorial_projectors(5 * np.pi / 8)],
        [equatorial_projectors(np.pi / 8), equatorial_projectors(5 * np.pi / 8)],
    ]
    return quantum_model(phi_plus, settings)


class TestBellModels:
    def setup_method(self):
        self.sc = bell_scenario()

    def test_pr_box_cf(self):
        pr_box = EmpiricalModel(self.sc, {0: CORR, 1: CORR, 2: CORR, 3: ANTI})
        assert contextual_fraction(pr_box) == pytest.approx(1.0, abs=ATOL)

    def test_bell_table_cf(self):
        assert contextual_fraction(bell_table_model(self.sc)) == pytest.approx(
            0.25, abs=ATOL)

    def test_uniform_model_cf(self):
        assert contextual_fraction(uniform_model(self.sc)) == pytest.approx(
            0.0, abs=ATOL)


class TestQuantumModels:
    def setup_method(self):
        self.sc = bell_scenario()

    def test_bell_state_reproduces_table_i(self):
        """|Phi+> at angles (0, pi/3) reproduces the Bell/CHSH table vector."""
        phi_plus = np.zeros(4, dtype=complex)
        phi_plus[0] = phi_plus[3] = 1 / np.sqrt(2)
        settings = [
            [equatorial_projectors(0.0), equatorial_projectors(np.pi / 3)],
            [equatorial_projectors(0.0), equatorial_projectors(np.pi / 3)],
        ]
        qm = quantum_model(phi_plus, settings)
        assert np.max(np.abs(qm.vector - bell_table_model(self.sc).vector)) < 1e-9

    def test_tsirelson_model_cf(self):
        """Tsirelson angles (pi/8, 5pi/8) both parties -> CF = sqrt(2) - 1."""
        assert contextual_fraction(tsirelson_model(self.sc)) == pytest.approx(
            np.sqrt(2) - 1, abs=ATOL)

    def test_ghz3_mermin_cf(self):
        """GHZ(3) with Pauli X, Y measurements -> CF = 1 (Mermin)."""
        ghz_settings = [
            [equatorial_projectors(0.0), equatorial_projectors(np.pi / 2)]
            for _ in range(3)
        ]
        ghz_model = quantum_model(ghz_state(3), ghz_settings)
        assert contextual_fraction(ghz_model) == pytest.approx(1.0, abs=ATOL)


class TestBellInequalityWitness:
    """Theorem 1: the dual LP returns a genuine Bell inequality whose
    normalised violation equals the model's CF."""

    def setup_method(self):
        self.sc = bell_scenario()

    def _check(self, model):
        a, _, violation = bell_inequality(model)
        assert is_bell_inequality(model, a)
        assert violation == pytest.approx(contextual_fraction(model), abs=ATOL)

    def test_bell_table_witness(self):
        self._check(bell_table_model(self.sc))

    def test_tsirelson_witness(self):
        self._check(tsirelson_model(self.sc))


class TestNoisyPRBox:
    """Noisy PR box: CF(eta) = max(0, 1 - 2*eta), threshold eta* = 1/2."""

    @pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75])
    def test_noise_sweep(self, eta):
        sc = bell_scenario()
        pr_box = EmpiricalModel(sc, {0: CORR, 1: CORR, 2: CORR, 3: ANTI})
        flat = uniform_model(sc)
        noisy = pr_box.mix(flat, 1 - eta)   # (1-eta) PR + eta uniform
        expected = max(0.0, 1 - 2 * eta)
        assert contextual_fraction(noisy) == pytest.approx(expected, abs=ATOL)
