"""Convert human annotation CSVs and attach three references to saved captions."""
import argparse
import csv
import json
import posixpath
from collections import Counter
from pathlib import Path

from utils.config import HUMAN_CAPTIONS_DIR
from utils.utils import atomic_write_json, load_records


def dataset_name(value):
    name = str(value).strip().lower().replace('-', '').replace('_', '')
    if name not in {'cvqa', 'gdvcr', 'geode'}:
        raise ValueError(f'Unknown dataset: {value!r}')
    return name


def image_key(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing image path or filename')
    return posixpath.normpath(value.strip().replace('\\', '/'))


def load_annotations(source, dataset=None):
    source = Path(source)
    files = sorted(source.glob('*.csv')) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f'No annotation CSVs in {source}')
    annotations = []
    seen = {}
    for path in files:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for number, raw in enumerate(csv.DictReader(stream), 2):
                row = {key.strip().lower(): value for key, value in raw.items() if key is not None}
                try:
                    name = dataset_name(row.get('dataset', ''))
                    if dataset and name != dataset:
                        continue
                    columns = ('human1 caption', 'human2 caption', 'human3 caption')
                    if not all(key in row for key in columns):
                        columns = ('ann1', 'ann2', 'ann3')
                    references = [row.get(key) for key in columns]
                    if not all(isinstance(text, str) and text.strip() for text in references):
                        raise ValueError('Expected three non-empty human captions')
                    references = [text.strip() for text in references]
                    image = image_key(row.get('imagepath') or row.get('path') or row.get('imagefilename'))
                    identity = (name, image)
                    if identity in seen:
                        raise ValueError(f'Duplicate annotation for {identity}: also in {seen[identity]}')
                    seen[identity] = f'{path.name}:{number}'
                    annotations.append({
                        'dataset': name, 'image_path': image,
                        'reference_caption': references,
                        'reference_source': {'file': path.name, 'row': number},
                    })
                except ValueError as exc:
                    raise ValueError(f'{path}:{number}: {exc}') from exc
    if not annotations:
        raise ValueError('No annotations found for the selected dataset')
    return annotations


def attach_references(records, annotations, dataset, image_field=None, annotated_only=False):
    annotations = [row for row in annotations if row['dataset'] == dataset]
    by_filename = {}
    for annotation in annotations:
        filename = posixpath.basename(annotation['image_path'])
        by_filename.setdefault(filename, []).append(annotation)
    samples, unmatched, used = [], [], set()
    fields = [image_field] if image_field else ['image_path', 'image', 'img_fn', 'image_file', 'ImagePath', 'ImageFileName', 'path']
    for index, record in enumerate(records):
        row = dict(record)
        if row.get('dataset') and dataset_name(row['dataset']) != dataset:
            raise ValueError(f'Record {index}: dataset differs from --dataset {dataset}')
        value = next((row[key] for key in fields if isinstance(row.get(key), str) and row[key].strip()), None)
        image = image_key(value)
        candidates = by_filename.get(posixpath.basename(image), [])
        exact = [candidate for candidate in candidates
                 if image == candidate['image_path'] or image.endswith('/' + candidate['image_path'])]
        candidates = exact or candidates
        if len(candidates) > 1:
            raise ValueError(f'Record {index}: ambiguous image {image!r}; use unique relative image paths')
        if not candidates:
            unmatched.append({'sample_index': index, 'image_path': image})
            continue
        annotation = candidates[0]
        refs = annotation['reference_caption']
        existing = row.get('reference_caption')
        if isinstance(existing, str) and existing.strip().startswith('['):
            existing = json.loads(existing)
        if existing not in (None, '') and existing != refs:
            raise ValueError(f'Record {index}: existing references conflict with the annotation CSV')
        row['image_path'] = value
        row['reference_caption'] = refs
        row['reference_source'] = annotation['reference_source']
        samples.append(row)
        used.add(annotation['image_path'])
    if unmatched and not annotated_only:
        raise ValueError(f'{len(unmatched)} prediction records have no human annotation; '
                         f'first: {unmatched[0]["image_path"]!r}. '
                         'Use --annotated-only to explicitly select the annotated subset.')
    if not samples:
        raise ValueError('No prediction images matched the human annotations')
    return samples, {
        'input_records': len(records), 'matched_records': len(samples),
        'matched_images': len(used), 'annotation_images': len(annotations),
        'unmatched_predictions': unmatched,
        'annotations_without_predictions': [row['image_path'] for row in annotations if row['image_path'] not in used],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=HUMAN_CAPTIONS_DIR,
                        help='Annotation CSV or directory; defaults to the bundled human annotations')
    parser.add_argument('--dataset', choices=['cvqa', 'gdvcr', 'geode'])
    parser.add_argument('--predictions', type=Path, help='Saved caption JSON or CSV to enrich')
    parser.add_argument('--image-field', help='Image path/filename column in predictions')
    parser.add_argument('--annotated-only', action='store_true', help='Explicitly discard predictions without human references')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON; pass this file to eval.evaluate')
    args = parser.parse_args(argv)
    if args.predictions and not args.dataset:
        parser.error('--dataset is required with --predictions')
    if not args.predictions and (args.annotated_only or args.image_field):
        parser.error('--annotated-only and --image-field require --predictions')
    if args.output.suffix.lower() != '.json':
        parser.error('--output must end in .json')
    if any(path and path.resolve() == args.output.resolve() for path in (args.annotations, args.predictions)):
        parser.error('Output must differ from the input files')
    return args


def main(argv=None):
    args = parse_args(argv)
    annotations = load_annotations(args.annotations, args.dataset)
    if args.predictions:
        records = load_records(args.predictions)
        samples, summary = attach_references(records, annotations, args.dataset,
                                              args.image_field, args.annotated_only)
        output = {}
        if args.predictions.suffix.lower() != '.csv':
            original = json.loads(args.predictions.read_text(encoding='utf-8'))
            if isinstance(original, dict):
                output = original.copy()
                if original.get('dataset') and dataset_name(original['dataset']) != args.dataset:
                    raise ValueError('Prediction file dataset differs from --dataset')
        output.update(dataset=args.dataset, samples=samples, reference_summary=summary)
    else:
        # Paths from the original CSVs may be relative or filenames only.
        # Joining to predictions preserves their usable image paths.
        summary = {'num_images': len(annotations),
                   'per_dataset': dict(Counter(row['dataset'] for row in annotations))}
        output = {'samples': annotations, 'reference_summary': summary}
        if args.dataset:
            output['dataset'] = args.dataset
    atomic_write_json(args.output, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
