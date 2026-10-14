import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import posixpath
import random

from utils.utils import (
    atomic_write_json,
    load_records,
    normalize_image_path,
    stable_country_seed,
)
from utils.config import VALIDATION_RATIO, SPLIT_OUTPUT_DIR, DEFAULT_SEED, TRAIN_SPLIT, VAL_SPLIT


def input_files(source):
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    # The positive generator writes both a combined file and per-country copies.
    combined = source / '_all_captioned.json'
    if combined.is_file():
        return [combined]
    paths = sorted(p for p in source.glob('*.json') if not p.name.startswith('_'))
    if not paths:
        raise ValueError(f'No caption JSON files directly under {source}')
    return paths


def country_of(row):
    country = row.get('country')
    if not isinstance(country, str) or not country.strip():
        country = row.get('selected_country')
    if not isinstance(country, str) or not country.strip():
        raise ValueError('Each record needs country or selected_country metadata')
    return country.strip()


def split_records(records, val_ratio=VALIDATION_RATIO, seed=DEFAULT_SEED, require_negative=False):
    """Keep connected records sharing an image path OR id in the same split.

    The ratio applies to independent groups within each country. Countries with
    only one group remain in training; other countries appear in both splits.
    """
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be strictly between 0 and 1')
    if not records:
        raise ValueError('No records to split')
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    countries, image_keys = [], []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f'Record {index}: expected an object')
        caption = row.get('generated_caption') or row.get('caption')
        for field, value in [('image', row.get('image')), ('caption', caption)]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Record {index}: missing/invalid {field}')
        if require_negative:
            negative = row.get('negative_caption')
            if not isinstance(negative, str) or not negative.strip():
                raise ValueError(f'Record {index}: missing/invalid negative_caption')
        countries.append(country_of(row))
        image = posixpath.normpath(normalize_image_path(row['image']))
        image_keys.append(image)
        keys = [('image', image)]
        sample_id = row.get('id')
        if sample_id is not None and str(sample_id).strip():
            keys.append(('id', str(sample_id).strip()))
        for key in keys:
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index

    groups = defaultdict(list)
    for index in range(len(records)):
        groups[root(index)].append(index)
    by_country = defaultdict(list)
    for indices in groups.values():
        labels = {countries[index] for index in indices}
        if len(labels) != 1:
            raise ValueError(f'Shared image/id has conflicting country labels: {sorted(labels)}')
        by_country[countries[indices[0]]].append(indices)

    train_indices, val_indices, singletons = [], [], []
    for country in sorted(by_country):
        country_groups = sorted(by_country[country],
                                key=lambda indices: min(image_keys[i] for i in indices))
        country_seed = stable_country_seed(seed, country)
        random.Random(country_seed).shuffle(country_groups)
        n = len(country_groups)
        n_val = min(n - 1, max(1, int(n * val_ratio + 0.5))) if n > 1 else 0
        if n == 1:
            singletons.append(country)
        for position, indices in enumerate(country_groups):
            (val_indices if position < n_val else train_indices).extend(indices)
    if not val_indices:
        raise ValueError('Validation would be empty: need at least two independent groups in one country')

    train = [records[i] for i in sorted(train_indices)]
    val = [records[i] for i in sorted(val_indices)]
    train_counts = Counter(countries[i] for i in train_indices)
    val_counts = Counter(countries[i] for i in val_indices)
    summary = {
        'seed': seed, 'requested_val_ratio': val_ratio,
        'actual_val_record_ratio': len(val) / len(records),
        'require_negative': require_negative,
        'grouping': 'connected records sharing normalized image path or non-empty id',
        'stratification': 'country; singleton groups kept in training',
        'records': {'total': len(records), 'train': len(train), 'val': len(val)},
        'groups': len(groups), 'countries_only_in_train': singletons,
        'per_country': {country: {
            'train': train_counts[country],
            'val': val_counts[country],
            'groups': len(by_country[country]),
        } for country in sorted(by_country)},
    }
    return train, val, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='Caption JSON file or directory of per-country JSON files (not recursive).')
    parser.add_argument('--output-dir', type=Path, default=SPLIT_OUTPUT_DIR)
    parser.add_argument('--val-ratio', type=float, default=VALIDATION_RATIO)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--require-negative', action='store_true',
                        help='Require negative_caption on every record for a common CAPS/Prompt/Prefix dataset.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    files = input_files(args.input)
    outputs = [args.output_dir / name for name in
               (f'{TRAIN_SPLIT}.json', f'{VAL_SPLIT}.json', '_split_manifest.json')]
    if any(path.resolve() == source.resolve() for path in outputs for source in files):
        raise ValueError('Output must not replace an input file; choose another --output-dir')
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError('Split output already exists; use another --output-dir or --overwrite')
    records = [row for path in files for row in load_records(path)]
    train, val, summary = split_records(records, args.val_ratio, args.seed, args.require_negative)
    summary['inputs'] = [str(path) for path in files]
    for path, data in zip(outputs, (train, val, summary)):
        atomic_write_json(path, data)
    print(json.dumps(summary['records'], indent=2))
    if summary['countries_only_in_train']:
        print('Countries with one independent group (train only): ' + ', '.join(summary['countries_only_in_train']))
    print(f'Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
