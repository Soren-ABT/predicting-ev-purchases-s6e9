"""Shared contracts for the champion reproduction package.

Everything here is defensive on purpose: this package exists to prove that one
specific CSV can be rebuilt byte for byte, so any silent coercion would defeat
its only purpose. Contract violations raise instead of being repaired.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = ROOT / 'inputs'
OUTPUT = ROOT / 'output'

ID = 'id'
TARGET = 'Will_Buy_EV'
COLUMNS = [ID, TARGET]
ROWS = 286571


class ContractError(RuntimeError):
    """A pinned input, an intermediate value, or an output broke its contract."""


def rank(values):
    """Average percentile rank in (0, 1].

    This single transformation is the whole algorithmic content of the final
    blend. It is monotone per column, which is why blending has to happen in
    rank space rather than on the raw probabilities.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ContractError('rank() expects a 1-D array')
    if not np.isfinite(values).all():
        raise ContractError('cannot rank an array containing non-finite values')
    return rankdata(values, method='average') / len(values)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned():
    """Read pinned.json, the machine-readable description of every input."""
    path = ROOT / 'pinned.json'
    if not path.exists():
        raise ContractError(f'missing contract file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def verify_fingerprint(path, expected_sha256, label):
    """Check a file against its recorded SHA256 before anything reads it."""
    path = Path(path)
    if not path.exists():
        raise ContractError(
            f'{label}: missing file {path}\n'
            f'  run "python src/fetch_sources.py" to obtain third-party inputs'
        )
    actual = sha256(path)
    if actual != expected_sha256:
        raise ContractError(
            f'{label}: SHA256 mismatch for {path}\n'
            f'  expected {expected_sha256}\n'
            f'  actual   {actual}\n'
            f'  this file is not the version this package was pinned to; refusing to continue'
        )
    return actual


def read_submission(path, expected_sha256=None, label=None, expect_rows=ROWS):
    """Read one `id,Will_Buy_EV` CSV and return (ids, predictions).

    Mirrors how the original export read its inputs: float_precision='round_trip'
    so the ranks are computed from the exact decimal values in the file rather
    than from a shortened float representation.
    """
    label = label or Path(path).name
    if expected_sha256 is not None:
        verify_fingerprint(path, expected_sha256, label)

    frame = pd.read_csv(path, float_precision='round_trip')
    if frame.columns.tolist() != COLUMNS:
        raise ContractError(f'{label}: expected columns {COLUMNS}, got {frame.columns.tolist()}')
    if len(frame) != expect_rows:
        raise ContractError(f'{label}: expected {expect_rows} rows, got {len(frame)}')
    if not frame[ID].is_unique:
        raise ContractError(f'{label}: duplicate ids')

    ids = frame[ID].to_numpy()
    predictions = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ContractError(f'{label}: predictions contain non-finite values')
    if predictions.min() < 0.0 or predictions.max() > 1.0:
        raise ContractError(
            f'{label}: predictions outside [0, 1] '
            f'(min {predictions.min()}, max {predictions.max()})'
        )
    return ids, predictions


def assert_same_ids(reference, other, reference_label, other_label):
    """Every pinned input must share one id order; the blend is positional."""
    if not np.array_equal(reference, other):
        raise ContractError(
            f'{other_label}: id column does not match {reference_label} elementwise.\n'
            f'  the blend is positional, so a reordered or subset file would silently '
            f'produce a wrong submission'
        )


def write_submission(path, ids, predictions):
    """Write an `id,Will_Buy_EV` CSV in the exact form the original export used."""
    path = Path(path)
    predictions = np.asarray(predictions, dtype=float)
    if len(predictions) != len(ids):
        raise ContractError('id and prediction lengths differ')
    if not np.isfinite(predictions).all():
        raise ContractError('refusing to write non-finite predictions')
    if predictions.min() < 0.0 or predictions.max() > 1.0:
        raise ContractError('refusing to write predictions outside [0, 1]')

    data = pd.DataFrame({ID: ids, TARGET: predictions}).to_csv(
        index=False, lineterminator='\n'
    ).encode()

    if path.exists():
        if path.read_bytes() != data:
            raise ContractError(
                f'{path} already exists with different content; refusing to overwrite.\n'
                f'  delete it yourself if you really want it regenerated'
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    # Read back and confirm the file reproduces the in-memory vector exactly.
    back = pd.read_csv(path, float_precision='round_trip')
    if back.columns.tolist() != COLUMNS or not np.array_equal(back[ID].to_numpy(), ids):
        raise ContractError(f'{path}: written file does not round-trip its id column')
    if not np.array_equal(back[TARGET].to_numpy(), predictions):
        raise ContractError(f'{path}: written file does not round-trip its predictions')
    return data
