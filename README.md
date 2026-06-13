# Contextual Games

Computational tools for studying contextuality, non-local games, and quantum advantage within the sheaf-theoretic framework developed by Abramsky and collaborators.

## Project aim

This project builds a computational pipeline for small non-local games. The pipeline will compute:

- the classical value by exhaustive search over deterministic strategies;
- the no-signalling value by linear programming;
- the quantum value using semidefinite programming relaxations from the NPA hierarchy;
- the contextual fraction of empirical models using linear programming.

The goal is to catalogue small games and analyse the relationship between contextuality, quantum advantage, game hardness, and noise robustness.

## Current status

Phase 0: infrastructure and validation.

Currently implemented:

- measurement scenarios;
- global assignments and local assignments;
- incidence matrix construction;
- empirical models;
- contextual fraction linear programme;
- dual Bell inequality extraction;
- quantum empirical models for binary-outcome projective measurements;
- validation script for known examples.

Validated examples include:

- PR box;
- Bell/CHSH table;
- Tsirelson CHSH model;
- GHZ-Mermin model;
- noisy PR box.

## Repository structure

```text
contextual-games/
├── src/contextual_games/     # Main Python package
├── scripts/                  # Validation and execution scripts
├── tests/                    # pytest tests
├── data/                     # Generated catalogues and outputs
├── notebooks/                # Exploratory notebooks
└── docs/                     # Notes, roadmap, and validation records
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

## Phase 0 validation

Run:

```bash
python scripts/validate_phase0.py
```

## Development notes

This repository is being developed as research software for an MSc project on contextuality and non-local games.

The emphasis is on:

- correctness before scale;
- validation against known examples;
- reproducible computations;
- clear separation between mathematical objects, optimisation routines, and generated catalogues.

## Roadmap

### Phase 0 — Infrastructure and validation

Build and validate the basic sheaf-theoretic machinery:

- scenarios;
- empirical models;
- incidence matrices;
- contextual fraction LP;
- Bell inequality dual.

### Phase 1 — Classical and no-signalling values

Implement non-local games and compute:

- classical value;
- no-signalling value;
- validation on CHSH and GHZ games.

### Phase 2 — Enumeration and symmetry reduction

Enumerate small binary-outcome games and deduplicate up to symmetry equivalence.

### Phase 3 — Quantum values

Estimate quantum values using SDP relaxations, beginning with low-level NPA relaxations and XOR-game fallbacks.

### Phase 4 — Analysis

Study:

- the landscape of quantum advantage;
- tightness of contextual-fraction bounds;
- degradation under depolarising noise.

### Phase 5 — Write-up

Produce the dissertation and, optionally, a browsable library of contextual games.

