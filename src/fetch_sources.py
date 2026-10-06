"""Obtain the two third-party prediction files this package is pinned to.

These two files are deliberately not redistributed with this repository: they are
other people's public work, and the Kaggle dataset that hosts them carries its own
terms. This script fetches them at the exact version, then proves the bytes match
what the champion submission was built from.

Two routes:

    python src/fetch_sources.py                 # via the Kaggle CLI (needs auth)
    python src/fetch_sources.py --from-dir DIR  # from an already extracted copy

Neither route is allowed to succeed on a hash mismatch.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, INPUTS, load_pinned, sha256, verify_fingerprint  # noqa: E402

# Local name -> file name inside the source dataset.
THIRD_PARTY = {
    'M': ('inputs/megayak_r8_m00_mega_verbatim.csv', 'r8_m00_mega_verbatim.csv'),
    'C': ('inputs/defiaudit_r6_rebuilt_94651.csv', 'r6_rebuilt_94651.csv'),
}


def report(entries):
    """Print one aligned line per input."""
    for key, spec in entries:
        state = 'OK' if spec['ok'] else 'MISSING' if not spec['present'] else 'MISMATCH'
        print(f'  [{state:8}] {key:>2}  {spec["path"]}')
        if spec['ok']:
            print(f'             sha256 {spec["sha256"]}')
        elif spec['present']:
            print(f'             expected {spec["sha256"]}')
            print(f'             actual   {spec["actual"]}')


def inspect(pinned):
    """Check every pinned input's presence and fingerprint without changing anything."""
    results = []
    for entry in pinned['inputs']:
        key, relative = entry['key'], entry['file']
        path = pinned['_root'] / relative
        record = {'path': relative, 'sha256': entry['sha256'], 'present': path.exists(), 'ok': False}
        if record['present']:
            record['actual'] = sha256(path)
            record['ok'] = record['actual'] == entry['sha256']
            record['resolved'] = path
        results.append((key, record))
    return results


def copy_from_dir(source, pinned):
    """Copy the third-party files out of a local extracted dataset directory."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise ContractError(f'--from-dir is not a directory: {source}')

    index = {p.name: p for p in source.rglob('*') if p.is_file()}
    print(f'scanning {source} ({len(index)} files)')

    for entry in pinned['inputs']:
        key = entry['key']
        if key not in THIRD_PARTY:
            continue
        local_relative, source_name = THIRD_PARTY[key]
        found = index.get(source_name)
        if found is None:
            raise ContractError(
                f'{key}: {source_name} not found under {source}\n'
                f'  expected a copy of {entry["source"]["dataset"]} version '
                f'{entry["source"]["version"]}'
            )
        actual = sha256(found)
        if actual != entry['sha256']:
            raise ContractError(
                f'{key}: {found} has SHA256 {actual}\n'
                f'  expected {entry["sha256"]}\n'
                f'  this is a different version of the file; refusing to use it'
            )
        destination = pinned['_root'] / local_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, destination)
        print(f'  copied {source_name} -> {local_relative}')


def download_via_kaggle(pinned):
    """Attempt the Kaggle CLI route; return True only if both files landed verified."""
    cli = shutil.which('kaggle')
    if cli is None:
        print('kaggle CLI not found on PATH; skipping the download route.')
        return False

    entry = next(e for e in pinned['inputs'] if e['key'] == 'M')
    dataset, version = entry['source']['dataset'], entry['source']['version']
    with tempfile.TemporaryDirectory(prefix='s6e9-sources-') as staging:
        command = [cli, 'datasets', 'download', '-d', dataset, '-v', str(version),
                   '-p', staging, '--unzip']
        print('running:', ' '.join(command))
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.SubprocessError) as error:
            print(f'kaggle CLI invocation failed: {error}')
            return False
        if completed.returncode != 0:
            print(f'kaggle CLI exited {completed.returncode}')
            tail = (completed.stderr or completed.stdout or '').strip().splitlines()[-5:]
            for line in tail:
                print('  ' + line)
            return False
        try:
            copy_from_dir(staging, pinned)
        except ContractError as error:
            print(f'downloaded content failed verification: {error}')
            return False
    return True


def manual_instructions(pinned):
    print()
    print('Automatic retrieval did not complete. Fetch the files manually:')
    for entry in pinned['inputs']:
        if entry['key'] not in THIRD_PARTY:
            continue
        local_relative, source_name = THIRD_PARTY[entry['key']]
        print()
        print(f'  {entry["key"]}: {entry["source"]["dataset"]} version {entry["source"]["version"]}')
        print(f'      page : {entry["source"]["url"]}')
        print(f'      file : {source_name}')
        print(f'      save as : {local_relative}')
        print(f'      sha256  : {entry["sha256"]}')
    print()
    print('If you already have the dataset extracted somewhere, use:')
    print('  python src/fetch_sources.py --from-dir <that directory>')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--from-dir', help='copy the third-party files from an extracted dataset copy')
    parser.add_argument('--check', action='store_true', help='only report status, never download')
    arguments = parser.parse_args()

    pinned = load_pinned()
    pinned['_root'] = Path(__file__).resolve().parents[1]
    INPUTS.mkdir(parents=True, exist_ok=True)

    print('pinned inputs')
    report(inspect(pinned))

    if arguments.check:
        return 0

    if arguments.from_dir:
        copy_from_dir(arguments.from_dir, pinned)
    elif not all(record['ok'] for key, record in inspect(pinned) if key in THIRD_PARTY):
        if not download_via_kaggle(pinned):
            manual_instructions(pinned)
            return 1
    else:
        print('third-party inputs already present and verified; nothing to download.')

    print()
    print('final state')
    results = inspect(pinned)
    report(results)
    failed = [key for key, record in results if not record['ok']]
    if failed:
        print()
        print(f'INCOMPLETE: {", ".join(failed)} could not be verified.')
        for key in failed:
            if key not in THIRD_PARTY:
                print(f'  {key} ships with this repository; restore it from version control.')
        return 1
    print()
    print('All pinned inputs are present and match their recorded SHA256.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ContractError as error:
        print(f'\nCONTRACT ERROR\n{error}', file=sys.stderr)
        sys.exit(2)
