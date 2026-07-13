# Contextual Games

**How much of quantum advantage is contextuality?** This repository
contains the computational side of an MSc research project (UCL Quantum
Technologies, supervised by Prof. Samson Abramsky) investigating that
question for non-local games, in the sheaf-theoretic framework of
Abramsky–Brandenburger and the contextual fraction of Abramsky, Barbosa
and Mansfield (arXiv:1705.07918).

**The full write-up of methodology, findings, and open questions is in
[`docs/overview.pdf`](docs/overview.pdf).** This README is the short
version.

## The idea

A non-local game asks separated players, who cannot communicate, to
produce coordinated answers to a referee's questions. A strategy is a
table of outcome probabilities, one distribution per question context,
and the framework grades such tables by their **contextual fraction
CF**: the proportion of the behaviour that admits no classical
(pre-assigned-value) explanation. CF is computable by linear
programming, its dual yields the maximally violated Bell inequality as
a witness, and every behaviour splits into a classical part and a
strongly contextual part, e = NCF·e^NC + CF·e^SC.

Each game then has three values, over three nested sets of strategies:

- **ω_c** — the best classical (pre-agreed) strategy, by exhaustive
  search;
- **ω_q** — the best quantum (entanglement-assisted) strategy,
  *certified* by sandwiching between semidefinite-programming upper
  bounds and explicit qubit strategies as lower bounds;
- **ω_ns** — the best strategy any relativity-respecting theory could
  offer, by linear programming over the no-signalling polytope.

The project enumerates **every** small game up to relabelling symmetry
(a game is just a win/lose grid, hence an integer; symmetries are cell
permutations; a family's name is the smallest integer in its orbit)
and solves the whole landscape at all three levels.

## What it has found so far

- **All 65,536 two-party binary games collapse to 805 families.** Only
  9 admit any advantage beyond classical at all; CHSH is the unique
  maximal case. Post-classical structure is rare, and the textbook
  game is the summit of the landscape.
- **A game quantum mechanics declines to play.** One three-party XOR
  family has the maximal possible no-signalling gap (ω_c = 3/4,
  ω_ns = 1) and certified ω_q = 3/4 exactly: no-signalling advantage
  does not imply quantum advantage.
- **Pseudo-telepathy needs unasked questions.** No fully-constrained
  three-party XOR game admits a perfect quantum strategy; the GHZ
  perfect win survives only when half the referee's questions are
  unconstrained.
- **A resource bound splits cleanly.** The ABM bound
  p̄_F ≥ NCF·(n−k)/n is *exactly tight* at the quantum optimum for
  every XOR family computed, and strictly loose for every non-XOR
  family with quantum advantage — sixteen for sixteen. A two-line
  strengthening of the bound, p̄_F ≥ NCF·(1−ω_c) + CF·(1−ω_ns),
  explains the split (XOR games have ω_ns = 1) and is itself exactly
  tight in 11 of the 16 cases. Whether the strengthened form is known,
  and whether its saturation at XOR optima is provable, are the
  project's open questions.
- **The resource can outlive the advantage.** Under noise, the CHSH
  and Mermin strategies lose their contextuality and their usefulness
  at the same threshold (1 − 1/√2 and 1/2 respectively, recovering the
  known visibilities) — but for one family there is a band of noise
  rates in which the strategy remains certifiably contextual while no
  longer beating classical play.

Epistemic status per finding (computed / certified / reproduction of
known results / conjectured) is tracked explicitly in the overview
document; several findings are expected to be rediscoveries and are
claimed only as validation of the pipeline.

## Methodology in one line

**No component is used until it reproduces literature values.** Every
phase is gated on known anchors — the PR box, the Bell/CHSH tables, the
Tsirelson bound, GHZ–Mermin, the standard noise visibilities — and the
same checks form a 46-test regression suite. All data in `data/` is
derivable from source: delete it and the scripts below rebuild every
number and figure.

## Install

Requires Python 3.12+. The repository must live outside iCloud-synced
folders (Documents/Desktop) — sync interferes with editable installs.

```bash
git clone https://github.com/97wfinney/contextual-games.git
cd contextual-games
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q     # 46 passed = healthy
```

## Run

Validation gates (each asserts the known-value anchors for its phase):

```bash
.venv/bin/python scripts/validate_phase0.py   # CF machinery
.venv/bin/python scripts/validate_phase1.py   # game values
.venv/bin/python scripts/validate_phase2.py   # enumeration & symmetry
.venv/bin/python scripts/validate_phase3.py   # quantum bracket
.venv/bin/python scripts/validate_phase4.py   # tightness & noise
```

Full reproduction of all data and figures (~15 minutes; order matters):

```bash
.venv/bin/python scripts/build_catalogue.py           # classical/NS catalogues
.venv/bin/python scripts/compute_quantum_values.py    # certified quantum values
.venv/bin/python scripts/run_phase4.py                # tightness & noise analysis
.venv/bin/python scripts/plot_noise.py                # the noise figure
```

## Layout

src/contextual_games/   cfraction · games · enumeration · quantum · analysis

scripts/                validation gates and production runs

tests/                  46-test regression suite

data/                   generated catalogues (CSV) and figures

docs/                   overview.pdf 

## References

- S. Abramsky, R. S. Barbosa, S. Mansfield, *Contextual fraction as a
  measure of contextuality*, PRL 119, 050504 (2017); arXiv:1705.07918.
- S. Abramsky, A. Brandenburger, *The sheaf-theoretic structure of
  non-locality and contextuality*, NJP 13, 113036 (2011).
