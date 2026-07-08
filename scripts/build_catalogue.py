"""
build_catalogue.py -- Phase 2 production run.

Solves one representative of every game equivalence class and writes the
catalogue CSVs into data/:

    data/catalogue_2party_general.csv   all 805-ish general 2-party classes
    data/catalogue_xor2.csv             XOR games, n=2
    data/catalogue_xor3.csv             XOR games, n=3

Each row: canonical_id, family, orbit/count, n_parties, omega_c, k,
hardness, omega_ns, ns_gap, cf_ns_strategy.

Prints a summary at the end: the classes with the largest
no-signalling/classical gap (the interesting corner of the landscape).
"""

import os
import time

from contextual_games.enumeration import (
    bell_scenario, bits_from_id,
    enumerate_two_party_classes, enumerate_xor_classes, xor_game_bits,
    canonical_form, symmetry_group,
    solve_game_bits, build_catalogue,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def solve_family(label, scenario, class_iter, n_parties):
    rows = []
    t0 = time.time()
    for i, (cid, meta) in enumerate(class_iter):
        row = {"canonical_id": cid, "family": label,
               "orbit": meta, "n_parties": n_parties}
        row.update(solve_game_bits(scenario, bits_from_id(
            cid, len(scenario.local_assignments))))
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {label}: {i + 1} classes solved "
                  f"({time.time() - t0:.1f}s)")
    print(f"  {label}: {len(rows)} classes in {time.time() - t0:.1f}s")
    return rows


def main():
    # -- general two-party sweep --------------------------------------------
    print("enumerating general 2-party classes...")
    classes2 = enumerate_two_party_classes()
    sc2 = bell_scenario(2)
    rows2 = solve_family(
        "general-2party", sc2,
        ((cid, orbit) for cid, orbit in sorted(classes2.items())),
        n_parties=2,
    )
    path = build_catalogue(rows2, os.path.join(
        DATA_DIR, "catalogue_2party_general.csv"))
    print(f"  wrote {path}")

    # -- XOR families ---------------------------------------------------------
    for n in (2, 3):
        print(f"enumerating XOR classes, n={n}...")
        xor = enumerate_xor_classes(n)
        sc = bell_scenario(n)
        rows = solve_family(
            f"xor-{n}party", sc,
            ((cid, meta["count"]) for cid, meta in sorted(xor.items())),
            n_parties=n,
        )
        path = build_catalogue(rows, os.path.join(
            DATA_DIR, f"catalogue_xor{n}.csv"))
        print(f"  wrote {path}")

    # -- headline summary -------------------------------------------------
    print("\nlargest no-signalling/classical gaps (general 2-party):")
    top = sorted(rows2, key=lambda r: -r["ns_gap"])[:8]
    print(f"  {'canonical_id':>14} {'orbit':>6} {'omega_c':>8} "
          f"{'omega_ns':>9} {'ns_gap':>7} {'cf_ns':>6}")
    for r in top:
        print(f"  {r['canonical_id']:>14} {r['orbit']:>6} "
              f"{r['omega_c']:>8.4f} {r['omega_ns']:>9.4f} "
              f"{r['ns_gap']:>7.4f} {r['cf_ns_strategy']:>6.3f}")

    n_contextual = sum(1 for r in rows2 if r["ns_gap"] > 1e-9)
    print(f"\n  classes with omega_ns > omega_c: "
          f"{n_contextual} / {len(rows2)}")


if __name__ == "__main__":
    main()
