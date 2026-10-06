"""Rebuild the champion submission from its three pinned prediction files.

    python src/reproduce_champion.py

Reads M, C and O, applies the recorded blend, and asserts the result matches the
SHA256 the champion file was pinned to. A mismatch is a hard failure: this script
never writes a submission it cannot vouch for.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import (  # noqa: E402
    ContractError, OUTPUT, assert_same_ids, load_pinned, rank,
    read_submission, sha256, write_submission,
)


def blend(m, c, o, weights):
    """The recorded blend, performed entirely in average-percentile-rank space."""
    total = weights['M'] + weights['C'] + weights['O']
    if abs(total - 1.0) > 1e-12:
        raise ContractError(f'blend weights must sum to 1.0, got {total}')
    score = weights['M'] * rank(m) + weights['C'] * rank(c) + weights['O'] * rank(o)
    return rank(score)


def main():
    pinned = load_pinned()
    weights = {entry['key']: float(entry['weight']) for entry in pinned['inputs']}

    print(f'formula  {pinned["formula"]}')
    print()
    print('reading pinned inputs')

    vectors, ids, labels = {}, None, {}
    for entry in pinned['inputs']:
        key = entry['key']
        path = Path(__file__).resolve().parents[1] / entry['file']
        file_ids, predictions = read_submission(
            path, expected_sha256=entry['sha256'], label=f'{key} ({path.name})'
        )
        owner = {'third_party': f'third-party, {entry["contributor"]}',
                 'own': 'this project'}[entry['ownership']]
        print(f'  {key}  weight {weights[key]:.3f}  {path.name}')
        print(f'      {owner}  sha256 verified')
        if ids is None:
            ids, labels = file_ids, entry['file']
        else:
            assert_same_ids(ids, file_ids, labels, entry['file'])
        vectors[key] = predictions

    print()
    print('blending in average-percentile-rank space')
    predictions = blend(vectors['M'], vectors['C'], vectors['O'], weights)

    output_name = pinned['output']['filename']
    destination = OUTPUT / output_name
    print(f'writing {destination}')
    write_submission(destination, ids, predictions)

    actual = sha256(destination)
    expected = pinned['output']['sha256']
    print()
    print(f'  rows    {len(predictions)}')
    print(f'  range   [{predictions.min():.12g}, {predictions.max():.12g}]')
    print(f'  sha256  {actual}')
    print(f'  pinned  {expected}')

    if actual != expected:
        raise ContractError(
            'reproduced file does not match the pinned champion fingerprint.\n'
            '  the inputs verified against their hashes, so the blend itself differs '
            'from the one that produced the champion file — do not submit this output'
        )

    report = {
        'reproduced': output_name,
        'sha256': actual,
        'matches_pinned': True,
        'rows': len(predictions),
        'formula': pinned['formula'],
        'weights': weights,
        'inputs': [
            {'key': entry['key'], 'file': entry['file'], 'sha256': entry['sha256'],
             'ownership': entry['ownership'], 'contributor': entry['contributor'],
             'weight': weights[entry['key']]}
            for entry in pinned['inputs']
        ],
        'caveat': pinned['scores']['comparison'],
    }
    (OUTPUT / 'reproduction_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print()
    print('MATCH: the reproduced file is byte-identical to the pinned champion submission.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ContractError as error:
        print(f'\nCONTRACT ERROR\n{error}', file=sys.stderr)
        sys.exit(2)
