"""
plot_noise.py -- Q3 figure: how success and contextual fraction decay
with white noise, for two instructive classes:

  left  -- CHSH: advantage and resource die together at 1 - 1/sqrt(2);
  right -- the 3-party XOR class id 7595718147998062230: the resource
           (CF > 0) survives to eta ~ 0.44 but the advantage is gone by
           eta ~ 0.29 -- a noise band with contextuality but no
           advantage for this game.

Writes data/noise_sweeps.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contextual_games.enumeration import bits_from_predicate, bits_from_id
from contextual_games.analysis import (
    optimal_strategy_model, noise_sweep,
    advantage_threshold, cf_threshold,
)
from contextual_games.enumeration import bell_scenario, game_from_bits
from contextual_games.games import classical_value

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CASES = [
    ("CHSH", 2,
     bits_from_predicate(2, lambda x, o: (o[0] ^ o[1]) == (x[0] & x[1]))),
    ("3-party XOR class ...062230", 3,
     bits_from_id(7595718147998062230, 64)),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
etas = np.linspace(0, 0.6, 31)

for ax, (label, n, bits) in zip(axes, CASES):
    _, model = optimal_strategy_model(bits, n, restarts=6, seed=1)
    rows = noise_sweep(bits, n, model, etas)
    game = game_from_bits(bell_scenario(n), bits)
    omega_c, _, _ = classical_value(game)
    e_adv = advantage_threshold(bits, n, model)
    e_cf = cf_threshold(model)

    ax.plot(etas, [r["success"] for r in rows], label="success $p_S(\\eta)$")
    ax.plot(etas, [r["cf"] for r in rows], label="CF$(\\eta)$")
    ax.axhline(omega_c, ls=":", color="grey",
               label=f"$\\omega_c$ = {omega_c:.3f}")
    ax.axvline(e_adv, ls="--", color="tab:red", alpha=0.7,
               label=f"$\\eta_{{adv}}$ = {e_adv:.3f}")
    ax.axvline(e_cf, ls="--", color="tab:green", alpha=0.7,
               label=f"$\\eta_{{CF}}$ = {e_cf:.3f}")
    if e_cf > e_adv + 1e-3:
        ax.axvspan(e_adv, e_cf, color="orange", alpha=0.12)
    ax.set_xlabel("noise rate $\\eta$")
    ax.set_title(label)
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_ylabel("value")
fig.suptitle("Noise robustness: advantage vs resource")
fig.tight_layout()
out = os.path.join(DATA_DIR, "noise_sweeps.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
