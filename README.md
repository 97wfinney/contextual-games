# Contextual Games

**How does contextuality relate to quantum advantage in non-local games?**

This repository contains the computational work for an MSc research project in Quantum Technologies at UCL, supervised by Professor Samson Abramsky.

The project uses the sheaf-theoretic framework of Abramsky and Brandenburger and the contextual fraction introduced by Abramsky, Barbosa and Mansfield. It investigates small non-local games to understand when contextuality enables an advantage, how much contextuality a successful strategy requires, and how these relationships change under noise.

**Status:** this is an ongoing research repository. The implementation has been checked against standard examples and includes a regression test suite. Numerical results, error handling and their mathematical interpretation are still being reviewed. A fuller written overview is in preparation.

## The research questions

The current work centres on three questions:

1. **Game values:** which games allow quantum or no-signalling strategies to outperform classical strategies?
2. **Contextuality and performance:** when is the contextual-fraction bound on game performance tight?
3. **Noise:** does a strategy lose its contextuality at the same point that it loses its advantage in a particular game?

The aim is to combine systematic computation with mathematical explanations of the patterns found.

## The basic setup

In a non-local game, a referee sends a question to each of several separated players. Each player returns an answer without communicating with the others. A winning rule determines whether their combined answers succeed.

A strategy gives a probability distribution over the possible answers for each combination of questions. In this project, the referee chooses uniformly from the game's question contexts.

There are three relevant values for a game:

| Value | Allowed strategies | Computational method |
|---|---|---|
| Classical value, ω_c | Local responses coordinated through shared randomness | Exhaustive search over deterministic assignments |
| Quantum value, ω_q | Local measurements on a shared quantum state | Explicit strategies and semidefinite relaxation estimates |
| No-signalling value, ω_ns | Probability distributions whose marginal statistics do not depend on distant question choices | Linear programming |

These values satisfy:

$$
\omega_c \leq \omega_q \leq \omega_{ns}.
$$

The no-signalling set is a mathematical outer bound on quantum behaviours; its elements need not be physically realisable.

## Measuring contextuality

In a Bell scenario, a non-contextual model is a behaviour that admits a local hidden-variable explanation: it can be expressed as a mixture of deterministic assignments of outcomes to all measurements.

The **non-contextual fraction**, NCF, is the largest weight of such a local component that can be included in a decomposition of a behaviour. The **contextual fraction** is its complement:

$$
\mathrm{CF}(e)=1-\mathrm{NCF}(e).
$$

A contextual fraction of zero means that the behaviour is local. A contextual fraction of one means that it is strongly contextual.

The code computes these quantities by linear programming. The dual programme can also produce a Bell inequality whose normalised violation witnesses the contextual fraction.

A central bound from Abramsky, Barbosa and Mansfield relates contextuality to a strategy's average failure probability:

$$
p_F(e)\geq \mathrm{NCF}(e)(1-\omega_c).
$$

This says that the local fraction of a strategy must contribute at least its share of the minimum classical failure rate. The project investigates when this bound is attained and what accounts for any remaining slack.

## Scope of the current computation

The completed enumerations cover:

- All **65,536 two-party games** with two questions and two answers per party, with uniformly distributed question pairs.
- All full-context XOR games with two or three parties, again with two questions and two answers per party.

An XOR game specifies a required parity of the players' answers for each question context. “Full-context” means that every combination of questions carries a parity condition.

Games are grouped under party permutations, local question relabellings and local answer relabellings. These transformations preserve the game values.

Each equivalence class is represented by a canonical integer encoding of its winning table. These identifiers depend on the encoding convention used in the code.

The general three-party game space has not been exhaustively classified.

## Preliminary observations

### Two-party classification

The 65,536 two-party games reduce to **805 equivalence classes** under the chosen relabellings.

Of these, **nine** have a no-signalling value strictly above their classical value. The quantum searches also find an advantage for each of these nine classes. CHSH attains the largest classical-to-no-signalling gap in this catalogue.

These are computational classification results. Their relationship to existing classifications and Bell-polytope results is part of the literature review.

### Quantum-value calculations

For the nine two-party classes with a no-signalling gap and the two- and three-party XOR catalogues, the explicit-strategy values and semidefinite relaxation estimates agree closely numerically.

Two non-trivial three-party XOR classes have no detected quantum advantage despite having a no-signalling advantage. Their numerical quantum estimates agree with classical values of 7/8 and 3/4 respectively.

No non-trivial full-context three-party XOR class in the catalogue has a quantum value reaching one. The standard GHZ game provides a contrasting example when only four of the eight contexts carry parity constraints: a quantum strategy wins perfectly, while classical strategies cannot.

These numerical observations are distinguished from exact analytic proofs.

### Tightness of the contextual-fraction bound

At the strategies located for the XOR games examined, the original resource bound is approximately saturated. It has positive slack at the located advantaged strategies for the eight non-XOR two-party classes.

A direct consequence of the optimal non-contextual decomposition is the stronger bound

$$
p_F(e)\geq
\mathrm{NCF}(e)(1-\omega_c)
+
\mathrm{CF}(e)(1-\omega_{ns}).
$$

The reason is that success probability is linear: the local component cannot exceed the classical value, and the remaining no-signalling component cannot exceed the no-signalling value.

This explains why an advantaged strategy must have strictly positive slack in the original bound whenever ω_ns < 1.

For XOR games, ω_ns = 1, so the additional term vanishes. This does **not** establish saturation of the original bound. Understanding when an optimal quantum strategy attains it remains a research question.

The strengthened bound is an elementary consequence of the decomposition theorem; no claim of novelty is made here.

### Contextuality and noise

The noise analysis mixes a strategy with the uniform behaviour:

$$
e_\eta=(1-\eta)e+\eta u.
$$

Here η is the proportion of global white noise, rather than an independent noise rate applied separately to each player.

The calculations recover the standard coincidence of the contextuality and game-advantage thresholds for the CHSH and GHZ–Mermin benchmark strategies.

Other examples exhibit a range of noise levels where a strategy remains contextual but no longer beats the classical value of the game being played. This illustrates that contextuality and usefulness for a particular game are different properties.

Noise thresholds are properties of the chosen strategy. Different optimal strategies for the same game can have different noise robustness.

## Validation and numerical limitations

The current regression suite contains **46 tests**, covering standard examples and structural checks. Benchmarks include:

- The PR box and the uniform behaviour.
- The Bell/CHSH probability table from Abramsky, Barbosa and Mansfield.
- The Tsirelson-optimal CHSH strategy.
- GHZ–Mermin correlations and game values.
- Standard white-noise thresholds.
- Relabelling invariance and enumeration checks.
- Contextual-fraction LPs and their dual Bell witnesses.

Passing these checks provides evidence that the components work on the tested cases. It does not establish every result in the catalogue.

Classical values are obtained from finite deterministic win counts. LPs, SDPs and strategy searches use floating-point arithmetic. Agreement between their outputs is not, by itself, a rigorous enclosure or an exact proof of equality.

In particular, the existing CSV field `certified` records the implementation's numerical gap check. It should not be interpreted as an independently verified mathematical certificate. Solver-certificate validation, gap handling and empirical-model input checks are under review.

The quantum and Phase 4 catalogues contain 16 entries representing 15 distinct game classes, because CHSH appears in both the general two-party and XOR catalogues.

Fixed seeds are used in the searches, but numerical outputs and the particular strategies located can vary with the software environment. Contextual fractions and noise thresholds reported for located strategies should be read accordingly.

## Installation

Requires Python 3.11 or later. The commands below use a Unix-style shell.

```bash
git clone https://github.com/97wfinney/contextual-games.git
cd contextual-games
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Run the regression tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

The main dependencies are NumPy, SciPy, CVXPY and Matplotlib.

## Running the calculations

The validation scripts exercise the benchmark examples for each stage:

```bash
.venv/bin/python scripts/validate_phase0.py   # Contextual-fraction calculations
.venv/bin/python scripts/validate_phase1.py   # Classical and no-signalling values
.venv/bin/python scripts/validate_phase2.py   # Enumeration and relabelling
.venv/bin/python scripts/validate_phase3.py   # Quantum-value estimates
.venv/bin/python scripts/validate_phase4.py   # Tightness and noise analysis
```

To regenerate the main catalogues and noise figure, run these scripts in order:

```bash
.venv/bin/python scripts/build_catalogue.py
.venv/bin/python scripts/compute_quantum_values.py
.venv/bin/python scripts/run_phase4.py
.venv/bin/python scripts/plot_noise.py
```

Two further scripts investigate the contextual-fraction lower bound and search for alternative strategies with lower contextual fraction:

```bash
.venv/bin/python scripts/floor_analysis.py
.venv/bin/python scripts/multiseed_cf.py
```

Run these after the main catalogue and Phase 4 calculations, in the order shown. The production scripts write or overwrite their corresponding outputs in `data/`. Runtime depends on the machine and optimisation settings.

## Repository layout

```text
src/contextual_games/
    cfraction.py       Scenarios, empirical models, contextual fraction and witnesses
    games.py           Game definitions, classical and no-signalling values
    enumeration.py     Game encodings, relabellings and catalogue construction
    quantum.py         Qubit strategy searches and moment-matrix relaxations
    analysis.py        Strategy reconstruction, bound tightness and noise analysis

scripts/               Validation and calculation entry points
tests/                 Regression tests
data/                  Saved numerical catalogues and figures
```

A fuller written account of the framework, worked examples, results and open questions is in preparation.

## References

- S. Abramsky and A. Brandenburger, *The sheaf-theoretic structure of non-locality and contextuality*, New Journal of Physics **13**, 113036 (2011).
- S. Abramsky, R. S. Barbosa and S. Mansfield, *Contextual fraction as a measure of contextuality*, Physical Review Letters **119**, 050504 (2017). [arXiv:1705.07918](https://arxiv.org/abs/1705.07918).