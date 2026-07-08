"""tests/test_enumeration.py -- Phase 2 regression tests."""

import numpy as np
import pytest

from contextual_games.enumeration import (
    bell_scenario, symmetry_group, canonical_form, bits_from_id,
    bits_from_predicate, xor_game_bits,
    enumerate_two_party_classes, enumerate_xor_classes,
    solve_game_bits,
)
from contextual_games.games import classical_value, no_signalling_value
from contextual_games.enumeration import game_from_bits


@pytest.fixture(scope="module")
def G2():
    return symmetry_group(2)


@pytest.fixture(scope="module")
def chsh_bits():
    return bits_from_predicate(2, lambda x, o: (o[0] ^ o[1]) == (x[0] & x[1]))


class TestSymmetryGroup:
    def test_group_size_n2(self, G2):
        assert len(G2) == 128          # 2! * 2^2 * 2^4

    def test_group_size_n3(self):
        assert len(symmetry_group(3)) == 3072   # 3! * 2^3 * 2^6

    def test_canonical_form_is_orbit_invariant(self, G2, chsh_bits):
        base, _ = canonical_form(chsh_bits, G2)
        rng = np.random.default_rng(0)
        for _ in range(5):
            g = G2[rng.integers(len(G2))]
            assert canonical_form(chsh_bits[g], G2)[0] == base


class TestTwoPartySweep:
    @pytest.fixture(scope="class")
    @classmethod
    def classes(cls):
        return enumerate_two_party_classes()

    def test_partition_is_exact(self, classes):
        assert sum(classes.values()) == 1 << 16

    def test_class_count(self, classes):
        assert len(classes) == 805

    def test_chsh_class(self, G2, chsh_bits, classes):
        cid, _ = canonical_form(chsh_bits, G2)
        assert cid in classes and classes[cid] == 8
        row = solve_game_bits(bell_scenario(2), bits_from_id(cid, 16))
        assert row["omega_c"] == pytest.approx(0.75)
        assert row["omega_ns"] == pytest.approx(1.0, abs=1e-7)
        assert row["cf_ns_strategy"] == pytest.approx(1.0, abs=1e-6)


class TestXORFamilies:
    def test_n2_has_two_classes(self):
        xor2 = enumerate_xor_classes(2)
        assert len(xor2) == 2
        assert sum(v["count"] for v in xor2.values()) == 16

    def test_n3_has_five_classes(self):
        xor3 = enumerate_xor_classes(3)
        assert len(xor3) == 5
        assert sum(v["count"] for v in xor3.values()) == 256


class TestEmbedding:
    def test_padded_ghz(self):
        mermin = {(0, 0, 0): 0, (0, 1, 1): 1, (1, 0, 1): 1, (1, 1, 0): 1}
        bits = bits_from_predicate(
            3,
            lambda x, o: ((o[0] ^ o[1] ^ o[2]) == mermin[x]) if x in mermin
            else True,
        )
        game = game_from_bits(bell_scenario(3), bits)
        wc, _, _ = classical_value(game)
        assert wc == pytest.approx(7 / 8)
        assert no_signalling_value(game) == pytest.approx(1.0, abs=1e-7)
