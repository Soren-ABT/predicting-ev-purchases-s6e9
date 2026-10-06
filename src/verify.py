"""Independently re-derive the champion submission and check the shipped artifact.

    python src/verify.py

This deliberately does not import reproduce_champion: it re-reads the pinned files
and recomputes the blend through a different ranking implementation (pandas rather
than scipy), so a mistake in one code path cannot hide behind the other.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    COLUMNS, ContractError, ID, INPUTS, OUTPUT, ROWS, TARGET,
    assert_same_ids, load_pinned, sha256,
)

CHECKS = []


def check(label, condition, detail=''):
    CHECKS.append((label, bool(condition), detail))
    print(f'  [{"PASS" if condition else "FAIL"}] {label}{("  " + detail) if detail else ""}')
    return bool(condition)


def pandas_rank(values):
    """Average percentile rank via pandas, independent of the scipy path."""
    return pd.Series(np.asarray(values, dtype=float)).rank(method='average', pct=True).to_numpy()


def main():
    pinned = load_pinned()
    root = Path(__file__).resolve().parents[1]
    weights = {entry['key']: float(entry['weight']) for entry in pinned['inputs']}

    print('1. pinned input fingerprints')
    raw = {}
    for entry in pinned['inputs']:
        path = root / entry['file']
        if not check(f'{entry["key"]}: {entry["file"]} present', path.exists()):
            raise ContractError(f'missing {entry["file"]}; run fetch_sources.py')
        digest = sha256(path)
        check(f'{entry["key"]}: sha256 matches pinned', digest == entry['sha256'], digest[:16] + '...')
        raw[entry['key']] = path

    print()
    print('2. input schema and id alignment')
    frames = {}
    for key, path in raw.items():
        frame = pd.read_csv(path, float_precision='round_trip')
        frames[key] = frame
        check(f'{key}: columns are {COLUMNS}', frame.columns.tolist() == COLUMNS)
        check(f'{key}: {ROWS} rows', len(frame) == ROWS, f'got {len(frame)}')
        check(f'{key}: ids unique', frame[ID].is_unique)
        values = frame[TARGET].to_numpy(dtype=float)
        check(f'{key}: predictions finite and within [0, 1]',
              np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1)

    reference_ids = frames['M'][ID].to_numpy()
    for key in ['C', 'O']:
        check(f'{key}: id order matches M', np.array_equal(reference_ids, frames[key][ID].to_numpy()))

    print()
    print('3. output artifact')
    destination = OUTPUT / pinned['output']['filename']
    if not check(f'{pinned["output"]["filename"]} exists', destination.exists(),
                 'run reproduce_champion.py first'):
        return 1
    digest = sha256(destination)
    check('output sha256 matches pinned', digest == pinned['output']['sha256'], digest[:16] + '...')

    produced = pd.read_csv(destination, float_precision='round_trip')
    check(f'output columns are {COLUMNS}', produced.columns.tolist() == COLUMNS)
    check(f'output has {ROWS} rows', len(produced) == ROWS, f'got {len(produced)}')
    check('output ids unique', produced[ID].is_unique)
    check('output id order matches inputs', np.array_equal(produced[ID].to_numpy(), reference_ids))
    observed = produced[TARGET].to_numpy(dtype=float)
    check('output predictions finite and within [0, 1]',
          np.isfinite(observed).all() and observed.min() >= 0 and observed.max() <= 1,
          f'[{observed.min():.6g}, {observed.max():.6g}]')

    print()
    print('4. independent recomputation (pandas ranking)')
    score = (weights['M'] * pandas_rank(frames['M'][TARGET])
             + weights['C'] * pandas_rank(frames['C'][TARGET])
             + weights['O'] * pandas_rank(frames['O'][TARGET]))
    recomputed = pandas_rank(score)
    check('recomputed vector equals the shipped output elementwise',
          np.array_equal(recomputed, observed),
          f'max abs diff {np.max(np.abs(recomputed - observed)):.3e}' if len(recomputed) == len(observed) else '')

    print()
    print('5. structural invariants of an average-percentile-rank vector')
    # Average ranks of tied values are half-integers, so with any tie present the
    # attainable values are k/(2n), not k/n. The residual ~1e-11 is float printing
    # precision at n = 286571, not a deviation from the grid.
    scaled = observed * (2 * ROWS)
    check('every value lies on the k/(2n) ranking grid (ties make half-integers reachable)',
          np.allclose(scaled, np.round(scaled), atol=1e-6),
          f'max deviation {np.max(np.abs(scaled - np.round(scaled))):.2e}')
    check('ranking the output again is the identity', np.array_equal(pandas_rank(observed), observed))
    check('weights sum to 1', abs(sum(weights.values()) - 1.0) < 1e-12,
          ' + '.join(f'{v:g}' for v in weights.values()))

    print()
    failed = [label for label, ok, _ in CHECKS if not ok]
    if failed:
        print(f'FAILED: {len(failed)} of {len(CHECKS)} checks did not pass.')
        for label in failed:
            print(f'  - {label}')
        return 1
    print(f'ALL {len(CHECKS)} CHECKS PASSED')
    print()
    print('Reproduced file is byte-identical to the pinned champion submission.')
    print(pinned['scores']['comparison'])
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ContractError as error:
        print(f'\nCONTRACT ERROR\n{error}', file=sys.stderr)
        sys.exit(2)
