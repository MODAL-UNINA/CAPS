"""Caption generation, automatic metrics and LLM judging for CVQA, GD-VCR, GeoDE."""
import argparse
import json
from pathlib import Path
from utils.utils import atomic_write_json, json_ready, load_records, load_rgb_image, resolve_image

from utils.config import (METHOD_FIELDS, METHOD_LABELS, CULTURAL_WORDS_FILE,
                          JUDGE_MODEL, OLLAMA_BASE_URL, LONGCLIP_MAX_CHARACTERS,
                          METRIC_NAMES, JUDGE_METRIC_NAMES)


def caption_for(row, method, field=None):
    if field:
        value = row.get(field)
    elif isinstance(row.get('captions'), dict) and method in row['captions']:
        value = row['captions'][method]
    else:
        value = row.get(METHOD_FIELDS[method])
    if not isinstance(value, str):
        raise ValueError(f'Missing caption for {method}; use --caption-field for a custom column')
    return value


def build_summary(samples, methods, args, errors, metadata):
    summary = dict(dataset=args.dataset, stage=args.stage, num_samples=len(samples),
                   num_failed=len(errors), methods={}, configuration=metadata)
    for method in methods:
        entry = {'method_name': METHOD_LABELS[method], 'num_captions': sum(method in row.get('captions', {}) for row in samples)}
        for section in ('metrics', 'judge'):
            values = [row[section][method] for row in samples if method in row.get(section, {})]
            if values:
                names = JUDGE_METRIC_NAMES if section == 'judge' else METRIC_NAMES
                entry[section] = {key: sum(value[key] for value in values)/len(values) for key in names}
                entry[f'num_{section}'] = len(values)
        summary['methods'][method] = entry
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['cvqa', 'gdvcr', 'geode'], required=True)
    p.add_argument('--method', choices=list(METHOD_FIELDS), required=True)
    p.add_argument('--stage', choices=['generate', 'metrics', 'judge', 'all'], default='all',
                   help='all = generate + automatic metrics; add --with-judge to include the judge')
    p.add_argument('--input', type=Path, required=True, help='HF save_to_disk directory, JSON, or CSV')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images-dir', type=Path, default=Path('.'))
    p.add_argument('--image-field')
    p.add_argument('--split')
    p.add_argument('--tag-field', default='tag')
    p.add_argument('--reference-field', default='reference_caption')
    p.add_argument('--caption-field')
    p.add_argument('--include-baseline', action='store_true')
    p.add_argument('--with-judge', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--model-name')
    p.add_argument('--device')
    p.add_argument('--load-in-4bit', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--caption-prompt')
    p.add_argument('--prompt-position', choices=['after_bos', 'prepend'])
    p.add_argument('--residual-scale', type=float)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-samples', type=int)
    p.add_argument('--cultural-words', type=Path, default=CULTURAL_WORDS_FILE, help='Union vocabulary (default: eval/all_cultural_words.txt)')
    p.add_argument('--longclip-repo', type=Path)
    p.add_argument('--longclip-checkpoint', type=Path)
    p.add_argument('--judge-model', default=JUDGE_MODEL)
    p.add_argument('--judge-base-url', default=OLLAMA_BASE_URL)
    p.add_argument('--judge-attempts', type=int, default=2)
    p.add_argument('--checkpoint-every', type=int, default=10)
    p.add_argument('--skip-errors', action='store_true', help='Explicitly skip failures; never turn errors into zero scores')
    args = p.parse_args(argv)
    generating = args.stage in ('generate', 'all')
    if generating and args.method != 'llava_baseline' and args.checkpoint is None:
        p.error('--checkpoint is required for tuned generation')
    if generating and args.caption_field:
        p.error('--caption-field applies to saved captions; use metrics/judge')
    if args.stage in ('metrics', 'all') and any(getattr(args, name) is None for name in (
            'cultural_words', 'longclip_repo', 'longclip_checkpoint')):
        p.error('Metrics require --cultural-words, --longclip-repo and --longclip-checkpoint')
    for name in ('max_new_tokens', 'max_samples', 'judge_attempts', 'checkpoint_every'):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    return args


def run(args):
    generating = args.stage in ('generate', 'all')
    scoring = args.stage in ('metrics', 'all')
    judging = args.stage == 'judge' or args.with_judge
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Input and output paths must differ')
    records = load_records(args.input, args.split)
    total = min(len(records), args.max_samples) if args.max_samples else len(records)
    if not total:
        raise ValueError('Input is empty')
    methods = [args.method] + (['llava_baseline'] if args.include_baseline and args.method != 'llava_baseline' else [])
    metadata = json_ready(vars(args))
    generator = scorer = judge = None
    if generating:
        from eval.inference import CaptionGenerator
        generator = CaptionGenerator(args)
        metadata['generation'] = generator.metadata
    if scoring:
        import torch
        from eval.auto_metrics import Metrics
        scorer = Metrics(args.cultural_words, args.longclip_repo, args.longclip_checkpoint)
        metadata['longclip_text_max_characters'] = LONGCLIP_MAX_CHARACTERS
    if judging:
        from eval.llm_judge import CaptionJudge
        judge = CaptionJudge(args.judge_model, args.judge_base_url, args.judge_attempts)
    output = args.output.resolve()
    summary_path = output.with_name(output.stem + '_summary.json')
    error_path = output.with_name(output.stem + '_errors.json')
    samples, errors = [], []

    def save():
        atomic_write_json(output, {'dataset': args.dataset, 'methods': methods, 'samples': samples})
        atomic_write_json(summary_path, build_summary(samples, methods, args, errors, metadata))
        atomic_write_json(error_path, errors)

    try:
        for index in range(total):
            try:
                row = dict(records[index])
                source = resolve_image(row, args.images_dir, args.image_field)
                image = load_rgb_image(source)
                item = json_ready(row)
                item['sample_index'] = index
                if isinstance(source, Path):
                    image_path = source.resolve()
                else:
                    image_path = output.parent / (output.stem + '_images') / f'{index:06d}.png'
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                item['image_path'] = str(image_path)
                item.pop('image', None)
                captions = generator.generate(image, args.max_new_tokens, args.include_baseline) if generating else {
                    method: caption_for(row, method, args.caption_field if method == args.method else None)
                    for method in methods}
                item['captions'] = captions
                item.pop('metrics', None)
                item.pop('judge', None)
                # Do not carry obsolete metric/judge dictionaries from legacy input.
                for key in list(item):
                    if key.startswith(('metrics_', 'llm_judge_')):
                        del item[key]
                if scoring:
                    if args.tag_field not in row or row[args.tag_field] is None:
                        raise ValueError(f'Missing RAM tags in {args.tag_field!r}')
                    item['metrics'] = {method: scorer.score(image, caption, row[args.tag_field]) for method, caption in captions.items()}
                if judging:
                    item['judge'] = {method: judge.score(image_path, caption, row.get(args.reference_field)) for method, caption in captions.items()}
                samples.append(item)
            except Exception as exc:
                errors.append({'sample_index': index, 'error': f'{type(exc).__name__}: {exc}'})
                if not args.skip_errors:
                    raise
                print(f'Skipped sample {index}: {exc}')
            if (index+1) % args.checkpoint_every == 0:
                save()
                print(f'Processed {index+1}/{total}; successful: {len(samples)}')
    finally:
        save()
    if not samples:
        raise RuntimeError(f'No samples evaluated; see {error_path}')
    print(json.dumps(build_summary(samples, methods, args, errors, metadata), indent=2))


if __name__ == '__main__':
    run(parse_args())
