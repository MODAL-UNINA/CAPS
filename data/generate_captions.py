from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import ijson
from ollama import Client
from tqdm import tqdm
from utils.config import (DEFAULT_IMAGES_ROOT, DEFAULT_SOURCE_JSON, POSITIVE_OUTPUT_DIR,
                          STYLE_WEIGHTS, CAPTION_STYLES, CAPTION_SYSTEM_PROMPT,
                          CAPTION_MODEL, OLLAMA_BASE_URL, DEFAULT_SEED, CULTURALGROUND_COUNTRIES)
from utils.utils import (
    atomic_write_json,
    normalize_image_path,
    build_positive_caption_prompt,
    stable_country_seed,
    clean_caption,
)


# Image formats sent directly to the vision backend.
DIRECT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Formats that are converted to PNG before captioning.
CONVERT_TO_PNG_EXTENSIONS = {".tif", ".tiff", ".webp", ".bmp", ".gif"}


def prepare_image_for_captioning(
    image_path: Path,
    cache_dir: Path,
) -> Path:
    """
    Return a vision-backend-safe image path.

    JPEG/PNG are used directly. TIFF/WEBP/BMP/GIF are converted to PNG and
    cached. This avoids failures caused by vision backends that do not decode
    all image formats reliably.
    """
    suffix = image_path.suffix.lower()

    if suffix in DIRECT_IMAGE_EXTENSIONS:
        return image_path

    if suffix not in CONVERT_TO_PNG_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format for captioning: {image_path.suffix}"
        )

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to convert TIFF/WEBP/BMP/GIF images. "
            "Install it with: pip install pillow"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the source directory tree so equal basenames cannot collide.
    relative = Path(*image_path.resolve().parts[1:])
    converted_path = cache_dir / relative.parent / (relative.name + ".png")
    converted_path.parent.mkdir(parents=True, exist_ok=True)

    if converted_path.is_file():
        return converted_path

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)

        # PNG supports RGB/RGBA. Convert palette, grayscale, CMYK, etc.
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        image.save(converted_path, format="PNG")

    return converted_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-json",
        type=Path,
        default=DEFAULT_SOURCE_JSON,
        help="Full CulturalGROUND JSON array.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=POSITIVE_OUTPUT_DIR,
    )
    parser.add_argument(
        "--target-per-country",
        type=int,
        default=100,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Large-source streaming/dedup.
    parser.add_argument(
        "--json-items-prefix",
        default="item",
        help=(
            "ijson prefix for samples. 'item' means the source is a top-level "
            "JSON array. Change only if your JSON has a wrapper object."
        ),
    )
    parser.add_argument(
        "--dedup-db",
        type=Path,
        default=None,
        help=(
            "Optional SQLite path used for memory-safe global deduplication. "
            "Default: <output-dir>/_dedup_seen.sqlite3"
        ),
    )
    parser.add_argument(
        "--dedup-commit-every",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--force-reselect",
        action="store_true",
        help="Ignore any saved selected subset and rebuild it from the full JSON.",
    )

    # Caption generation.
    parser.add_argument(
        "--model",
        default=CAPTION_MODEL,
    )
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--replacement-pool-multiplier",
        type=float,
        default=1.0,
        help=(
            "Extra same-country candidates reserved for replacing samples "
            "that cannot be captioned. 1.0 means one extra candidate per "
            "target sample."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Create the reproducible selected subset but do not caption it.",
    )
    parser.add_argument(
        "--max-caption-samples",
        type=int,
        default=None,
        help="Optional debugging limit applied only after selection.",
    )

    args = parser.parse_args()

    if args.target_per_country <= 0:
        parser.error("--target-per-country must be > 0.")
    if args.dedup_commit_every <= 0:
        parser.error("--dedup-commit-every must be > 0.")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be > 0.")
    if args.replacement_pool_multiplier < 0:
        parser.error("--replacement-pool-multiplier must be >= 0.")

    return args


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_for_match(value: str) -> str:
    """
    Normalize country names and image paths to underscore-delimited lowercase
    tokens. This makes e.g.:
        "South Korea" -> "south_korea"
        ".../South-Korea/foo.jpg" -> "..._south_korea_foo_jpg"
    """
    value = value.strip().lower().replace("\\", "/")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def build_country_alias_pairs() -> list[tuple[str, str]]:
    """Match official CulturalGround folder slugs to canonical country names."""
    pairs = [
        (normalize_for_match(country), country)
        for country in CULTURALGROUND_COUNTRIES
    ]
    pairs.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
    return pairs


def detect_country_from_image_path(
    raw_image_path: str,
    alias_pairs: list[tuple[str, str]],
) -> str | None:
    """
    Detect country only from the image path.

    Matching uses token boundaries after normalization, so alias "china" will
    match ".../china/image.jpg" but not an unrelated token containing "china".
    """
    normalized_path = normalize_for_match(raw_image_path)
    if not normalized_path:
        return None

    padded = f"_{normalized_path}_"

    matches: list[tuple[int, str]] = []
    for normalized_alias, canonical_country in alias_pairs:
        needle = f"_{normalized_alias}_"
        if needle in padded:
            matches.append((len(normalized_alias), canonical_country))

    if not matches:
        return None

    # Longest alias wins. If multiple aliases of same max length map to different
    # countries, skip as ambiguous rather than silently assigning one.
    max_len = max(length for length, _ in matches)
    top_countries = sorted({
        country for length, country in matches if length == max_len
    })
    if len(top_countries) != 1:
        return None

    return top_countries[0]


def normalized_dedup_image_path(raw_path: str) -> str:
    """
    Normalize path for duplicate detection without making it absolute.
    """
    path = normalize_image_path(raw_path)
    path = re.sub(r"/+", "/", path)
    return path.strip().lower()


def resolve_image_path(images_root: Path, raw_path: str) -> Path:
    path = Path(normalize_image_path(raw_path))
    return path if path.is_absolute() else images_root / path


def iter_source_samples(
    source_json: Path,
    items_prefix: str,
) -> Iterator[dict[str, Any]]:
    if not source_json.is_file():
        raise FileNotFoundError(f"Source JSON not found: {source_json}")

    with source_json.open("rb") as handle:
        for item in ijson.items(handle, items_prefix):
            if isinstance(item, dict):
                yield item


class SQLiteDeduplicator:
    """
    Memory-safe first-occurrence deduplication.

    A record is accepted only if BOTH its id and normalized image_path have never
    appeared before among eligible-country samples. UNIQUE constraints guarantee
    that a duplicate of either field is rejected.
    """

    def __init__(self, db_path: Path, commit_every: int = 10_000) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Rebuild for each fresh selection pass.
        if self.db_path.exists():
            self.db_path.unlink()

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=OFF;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("""
            CREATE TABLE seen (
                sample_id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL UNIQUE
            )
        """)
        self.conn.execute("BEGIN")

        self.commit_every = commit_every
        self.pending = 0

    def accept(self, sample_id: str, image_path: str) -> bool:
        try:
            self.conn.execute(
                "INSERT INTO seen(sample_id, image_path) VALUES (?, ?)",
                (sample_id, image_path),
            )
            accepted = True
        except sqlite3.IntegrityError:
            accepted = False

        self.pending += 1
        if self.pending >= self.commit_every:
            self.conn.commit()
            self.conn.execute("BEGIN")
            self.pending = 0

        return accepted

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()


def select_reproducible_subset(
    source_json: Path,
    output_dir: Path,
    target_per_country: int,
    replacement_pool_multiplier: float,
    seed: int,
    items_prefix: str,
    dedup_db: Path,
    dedup_commit_every: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
  
    canonical_countries = list(CULTURALGROUND_COUNTRIES)
    alias_pairs = build_country_alias_pairs()

    # We select a primary target pool plus deterministic reserve candidates.
    # The reserve is used only if a selected image cannot be captioned.
    pool_size = target_per_country + int(
        target_per_country * replacement_pool_multiplier
    )

    reservoirs: dict[str, list[dict[str, Any]]] = {
        country: [] for country in canonical_countries
    }
    eligible_unique_counts: dict[str, int] = {
        country: 0 for country in canonical_countries
    }
    path_match_counts: dict[str, int] = {
        country: 0 for country in canonical_countries
    }

    rng_by_country = {
        country: random.Random(stable_country_seed(seed, country))
        for country in canonical_countries
    }

    stats: dict[str, int] = {
        "streamed_samples": 0,
        "non_english_samples": 0,
        "matched_country_from_path": 0,
        "no_country_match": 0,
        "missing_id": 0,
        "missing_image_path": 0,
        "duplicate_id_or_image_path": 0,
        "eligible_unique_samples": 0,
    }

    deduper = SQLiteDeduplicator(
        dedup_db,
        commit_every=dedup_commit_every,
    )

    try:
        iterator = iter_source_samples(source_json, items_prefix)

        for sample in tqdm(
            iterator,
            desc="Streaming CulturalGROUND",
            unit="sample",
        ):
            stats["streamed_samples"] += 1

            # Keep English samples only, before country matching, deduplication
            # and reservoir sampling.
            language = str(sample.get("language", "")).strip().lower()
            if language != "en":
                stats["non_english_samples"] += 1
                continue

            raw_image = str(sample.get("image", "")).strip()
            if not raw_image:
                stats["missing_image_path"] += 1
                continue

            country = detect_country_from_image_path(
                raw_image,
                alias_pairs,
            )
            if country is None:
                stats["no_country_match"] += 1
                continue

            stats["matched_country_from_path"] += 1
            path_match_counts[country] += 1

            sample_id = str(sample.get("id", "")).strip()
            if not sample_id:
                stats["missing_id"] += 1
                continue

            dedup_path = normalized_dedup_image_path(raw_image)
            if not deduper.accept(sample_id, dedup_path):
                stats["duplicate_id_or_image_path"] += 1
                continue

            stats["eligible_unique_samples"] += 1
            eligible_unique_counts[country] += 1

            # Add explicit provenance of how country was obtained.
            enriched = dict(sample)
            enriched["selected_country"] = country
            enriched["country_detection_source"] = "image_path"

            reservoir = reservoirs[country]
            seen_n = eligible_unique_counts[country]
            rng = rng_by_country[country]

            if len(reservoir) < pool_size:
                reservoir.append(enriched)
            else:
                replacement_index = rng.randrange(seen_n)
                if replacement_index < pool_size:
                    reservoir[replacement_index] = enriched

    finally:
        deduper.close()

    selected_counts = {
        country: len(reservoirs[country])
        for country in canonical_countries
    }

    manifest = {
        "source_json": str(source_json),
        "country_registry": "utils.config.CULTURALGROUND_COUNTRIES",
        "target_per_country": target_per_country,
        "replacement_pool_multiplier": replacement_pool_multiplier,
        "replacement_pool_size_per_country": pool_size,
        "seed": seed,
        "json_items_prefix": items_prefix,
        "dedup_rule": (
            "global first-occurrence dedup among matched-country samples; "
            "reject if sample id OR normalized image path was already seen"
        ),
        "language_filter": "sample['language'] == 'en' (case-insensitive)",
        "country_detection_rule": (
            "country detected only from normalized image path using country "
            "names from the built-in 42-country list; longest unambiguous match wins"
        ),
        "sampling_rule": (
            "uniform reservoir sampling independently per country with "
            "byte-encoded deterministic per-country RNG"
        ),
        "caption_length_policy": (
            "adaptive; usually about 30-90 words, 1-3 sentences; no hard word bound"
        ),
        "image_format_policy": (
            "JPEG/PNG used directly; TIFF/TIF/WEBP/BMP/GIF converted to PNG "
            "before captioning"
        ),
        "replacement_policy": (
            "same-country reserve candidates are captioned only when a primary "
            "candidate fails; final output contains target_per_country successful "
            "captions whenever the reserve is sufficient"
        ),
        "countries": canonical_countries,
        "path_match_counts": path_match_counts,
        "eligible_unique_counts": eligible_unique_counts,
        "selected_counts": selected_counts,
        "stats": stats,
    }

    selected_dir = output_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    for country in canonical_countries:
        safe_name = normalize_for_match(country)
        atomic_write_json(
            selected_dir / f"{safe_name}.json",
            reservoirs[country],
        )

    combined = [
        sample
        for country in canonical_countries
        for sample in reservoirs[country]
    ]
    atomic_write_json(selected_dir / "_all_selected.json", combined)
    atomic_write_json(selected_dir / "_selection_manifest.json", manifest)

    return reservoirs, manifest


def load_saved_selection(
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected_dir = output_dir / "selected"
    manifest_path = selected_dir / "_selection_manifest.json"
    combined_path = selected_dir / "_all_selected.json"

    if not manifest_path.exists() or not combined_path.exists():
        raise FileNotFoundError("Saved selection files are incomplete.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combined = json.loads(combined_path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict) or not isinstance(combined, list):
        raise ValueError("Saved selection files are malformed.")

    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in combined:
        if not isinstance(sample, dict):
            continue
        country = str(sample.get("selected_country", "")).strip()
        if country:
            by_country[country].append(sample)

    return dict(by_country), manifest


def extract_qa(sample: dict[str, Any]) -> tuple[str, str] | None:
    question = answer = None

    for turn in sample.get("conversations", []):
        role = str(turn.get("from", "")).strip().lower()
        value = str(turn.get("value", "")).strip()

        if role == "human" and question is None:
            question = value.replace("<image>", "").strip().strip('"')
        elif role == "gpt" and answer is None:
            answer = value.strip().strip('"')

    return (question, answer) if question and answer else None


def choose_caption_style(sample: dict[str, Any]) -> tuple[str, str]:
    """
    Deterministic style assignment based on sample id + image path.
    """
    key = f"{sample.get('id', '')}::{sample.get('image', '')}"
    value = random.Random(int.from_bytes(key.encode("utf-8"), "big")).randrange(sum(weight for _, weight in STYLE_WEIGHTS))

    cumulative = 0
    for style_name, weight in STYLE_WEIGHTS:
        cumulative += weight
        if value < cumulative:
            return style_name, CAPTION_STYLES[style_name]

    name = STYLE_WEIGHTS[-1][0]
    return name, CAPTION_STYLES[name]


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""

    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", "")).strip()

    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "").strip()

    return str(response).strip()


def generate_caption(
    client: Client,
    model: str,
    image_path: Path,
    sample: dict[str, Any],
    country: str,
    temperature: float,
    max_retries: int,
) -> tuple[str, str]:
    qa = extract_qa(sample)
    if qa is None:
        raise ValueError("Missing/invalid QA.")

    question, answer = qa
    style_name, style_instruction = choose_caption_style(sample)

    user_text = build_positive_caption_prompt(question, answer, country, style_instruction)

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": CAPTION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_text,
                        "images": [str(image_path)],
                    },
                ],
                options={
                    "temperature": temperature,
                },
            )

            caption = clean_caption(extract_response_text(response))

            if not caption:
                raise ValueError("Model returned an empty caption.")

            # Only a sanity check, not a target-length constraint.
            if len(caption.split()) < 5:
                raise ValueError(
                    f"Caption implausibly short: {caption!r}"
                )

            return caption, style_name

        except Exception as exc:
            last_error = exc
            logging.warning(
                "Caption attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                image_path,
                exc,
            )

            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(
        f"Caption generation failed: {image_path}"
    ) from last_error


def load_existing_captioned(
    captioned_dir: Path,
    resume: bool,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    output_by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed_keys: set[str] = set()

    if not resume or not captioned_dir.exists():
        return output_by_country, processed_keys

    for path in sorted(captioned_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue

        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Cannot read existing caption file: %s", path)
            continue

        if not isinstance(records, list):
            continue

        for sample in records:
            if not isinstance(sample, dict):
                continue

            country = str(sample.get("selected_country", "")).strip()
            if not country:
                continue

            output_by_country[country].append(sample)

            key = f"{sample.get('id', '')}::{sample.get('image', '')}"
            if str(sample.get("generated_caption", "")).strip():
                processed_keys.add(key)

    return output_by_country, processed_keys


def save_caption_outputs(
    captioned_dir: Path,
    output_by_country: dict[str, list[dict[str, Any]]],
    processed_count: int,
    failed_count: int,
    total_expected: int,
) -> None:
    captioned_dir.mkdir(parents=True, exist_ok=True)

    all_captioned: list[dict[str, Any]] = []

    for country in sorted(output_by_country):
        records = output_by_country[country]
        safe_name = normalize_for_match(country)
        atomic_write_json(
            captioned_dir / f"{safe_name}.json",
            records,
        )
        all_captioned.extend(records)

    atomic_write_json(
        captioned_dir / "_all_captioned.json",
        all_captioned,
    )
    atomic_write_json(
        captioned_dir / "_caption_state.json",
        {
            "processed_samples": processed_count,
            "failed_samples": failed_count,
            "total_expected": total_expected,
            "complete": processed_count >= total_expected,
        },
    )


def caption_selected_samples(
    selected_by_country: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    """
    Caption the selected pool with robust resume + replacement behavior.

    - Existing generated captions are preserved and never regenerated.
    - Existing successful captions count toward target_per_country.
    - Uncaptioned primary candidates are tried first.
    - If one fails (missing/unreadable/unsupported image or model error),
      an unused same-country reserve candidate is tried.
    - TIFF/TIF/WEBP/BMP/GIF are converted to PNG before captioning.
    - The function keeps trying reserve candidates until each country reaches
      target_per_country successful captions or its reserve is exhausted.
    """
    captioned_dir = args.output_dir / "captioned"
    errors_path = captioned_dir / "_caption_errors.jsonl"
    image_cache_dir = args.output_dir / "_caption_image_cache"

    output_by_country, processed_keys = load_existing_captioned(
        captioned_dir,
        resume=args.resume,
    )

    # Keep only valid existing generated captions in the successful lists.
    successful_by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for country, records in output_by_country.items():
        seen_country_keys: set[str] = set()
        for record in records:
            key = f"{record.get('id', '')}::{record.get('image', '')}"
            if key in seen_country_keys:
                continue
            if str(record.get("generated_caption", "")).strip():
                successful_by_country[country].append(record)
                seen_country_keys.add(key)

    # The saved selection contains primary + reserve candidates.
    primary_by_country: dict[str, list[dict[str, Any]]] = {}
    reserve_by_country: dict[str, list[dict[str, Any]]] = {}

    for country in sorted(selected_by_country):
        pool = list(selected_by_country[country])
        primary_by_country[country] = pool[:args.target_per_country]
        reserve_by_country[country] = pool[args.target_per_country:]

    client = Client(host=args.base_url)
    failed_count = 0
    accepted_since_checkpoint = 0
    attempted_this_run = 0

    def sample_key(sample: dict[str, Any]) -> str:
        return f"{sample.get('id', '')}::{sample.get('image', '')}"

    def log_failure(
        *,
        country: str,
        sample: dict[str, Any],
        error_type: str,
        error: Exception | None = None,
        resolved_path: Path | None = None,
    ) -> None:
        nonlocal failed_count

        failed_count += 1

        payload: dict[str, Any] = {
            "type": error_type,
            "id": sample.get("id"),
            "selected_country": country,
            "image": sample.get("image"),
        }

        if resolved_path is not None:
            payload["resolved_path"] = str(resolved_path)

        if error is not None:
            payload["error"] = repr(error)

        append_jsonl(errors_path, payload)

    def try_caption(
        country: str,
        sample: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw_image = str(sample.get("image", "")).strip()

        if not raw_image:
            log_failure(
                country=country,
                sample=sample,
                error_type="missing_image_field",
            )
            return None

        image_path = resolve_image_path(args.images_root, raw_image)

        if not image_path.is_file():
            log_failure(
                country=country,
                sample=sample,
                error_type="missing_image_file",
                resolved_path=image_path,
            )
            return None

        try:
            caption_image_path = prepare_image_for_captioning(
                image_path=image_path,
                cache_dir=image_cache_dir,
            )
        except Exception as exc:
            log_failure(
                country=country,
                sample=sample,
                error_type="image_format_or_conversion_error",
                error=exc,
                resolved_path=image_path,
            )
            return None

        try:
            caption, style_name = generate_caption(
                client=client,
                model=args.model,
                image_path=caption_image_path,
                sample=sample,
                country=country,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
        except Exception as exc:
            logging.exception(
                "Captioning failed for %s",
                image_path,
            )
            log_failure(
                country=country,
                sample=sample,
                error_type="caption_generation_error",
                error=exc,
                resolved_path=image_path,
            )
            return None

        enriched = dict(sample)
        enriched["selected_country"] = country
        enriched["country_detection_source"] = "image_path"
        enriched["generated_caption"] = caption
        enriched["caption_style"] = style_name

        if image_path.suffix.lower() != caption_image_path.suffix.lower():
            enriched["caption_image_path"] = str(caption_image_path)
            enriched["caption_image_conversion"] = "converted_to_png"

        return enriched

    try:
        for country in sorted(primary_by_country):
            target = args.target_per_country

            # Existing captions count toward the target.
            successes = list(successful_by_country.get(country, []))
            existing_keys = {sample_key(s) for s in successes}

            # Build an ordered candidate queue: primary first, reserve second.
            # IMPORTANT: candidates already captioned in a previous run are
            # skipped before attempting them.
            candidates = (
                primary_by_country[country]
                + reserve_by_country[country]
            )

            for sample in candidates:
                if len(successes) >= target:
                    break

                if (
                    args.max_caption_samples is not None
                    and attempted_this_run >= args.max_caption_samples
                ):
                    break

                key = sample_key(sample)

                # This is the critical resume behavior:
                # already-captioned samples are never regenerated.
                if key in processed_keys or key in existing_keys:
                    continue

                attempted_this_run += 1

                result = try_caption(country, sample)

                if result is None:
                    # Do NOT add the failed sample to output.
                    # The next candidate (including a reserve candidate)
                    # becomes the replacement automatically.
                    continue

                result_key = sample_key(result)

                if result_key in existing_keys:
                    continue

                successes.append(result)
                existing_keys.add(result_key)
                processed_keys.add(result_key)

                output_by_country[country] = successes
                successful_by_country[country] = successes

                accepted_since_checkpoint += 1

                if accepted_since_checkpoint >= args.checkpoint_every:
                    save_caption_outputs(
                        captioned_dir=captioned_dir,
                        output_by_country=output_by_country,
                        processed_count=len(processed_keys),
                        failed_count=failed_count,
                        total_expected=sum(
                            min(
                                args.target_per_country,
                                len(v),
                            )
                            for v in primary_by_country.values()
                        ),
                    )
                    accepted_since_checkpoint = 0

            output_by_country[country] = successes

            if len(successes) < target:
                logging.warning(
                    "Country %s finished with %d/%d successful captions. "
                    "No more unused candidates were available.",
                    country,
                    len(successes),
                    target,
                )

    except KeyboardInterrupt:
        logging.warning(
            "Interrupted manually; saving caption checkpoint."
        )

    finally:
        save_caption_outputs(
            captioned_dir=captioned_dir,
            output_by_country=output_by_country,
            processed_count=len(processed_keys),
            failed_count=failed_count,
            total_expected=sum(
                min(
                    args.target_per_country,
                    len(v),
                )
                for v in primary_by_country.values()
            ),
        )

    print("\nCaptioning complete.")
    print(
        f"Successful captions: "
        f"{sum(len(v) for v in output_by_country.values())}"
    )
    print(f"New captioning errors: {failed_count}")
    print(f"Output: {captioned_dir}")

    for country in sorted(primary_by_country):
        count = len(output_by_country.get(country, []))
        print(
            f"{country:25s} "
            f"successful={count:5d}/{args.target_per_country}"
        )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dedup_db = (
        args.dedup_db
        if args.dedup_db is not None
        else args.output_dir / "_dedup_seen.sqlite3"
    )

    manifest_path = (
        args.output_dir
        / "selected"
        / "_selection_manifest.json"
    )

    if (
        args.resume
        and not args.force_reselect
        and manifest_path.exists()
    ):
        logging.info(
            "Loading previously saved reproducible selection from %s",
            manifest_path.parent,
        )
        selected_by_country, manifest = load_saved_selection(
            args.output_dir
        )

    else:
        logging.info(
            "Selecting samples from full CulturalGROUND file: %s",
            args.source_json,
        )

        selected_by_country, manifest = select_reproducible_subset(
            source_json=args.source_json,
            output_dir=args.output_dir,
            target_per_country=args.target_per_country,
            replacement_pool_multiplier=args.replacement_pool_multiplier,
            seed=args.seed,
            items_prefix=args.json_items_prefix,
            dedup_db=dedup_db,
            dedup_commit_every=args.dedup_commit_every,
        )

    print("\n" + "=" * 88)
    print("SELECTION SUMMARY")
    print("=" * 88)
    print(f"Seed: {manifest['seed']}")
    print(f"Target per country: {manifest['target_per_country']}")
    print(
        f"Replacement pool per country: "
        f"{manifest.get('replacement_pool_size_per_country', manifest['target_per_country'])}"
    )
    print("Language filter: language == 'en'")
    print(
        f"Non-English samples skipped: "
        f"{manifest['stats'].get('non_english_samples', 0)}"
    )
    print(
        f"Unique eligible samples: "
        f"{manifest['stats']['eligible_unique_samples']}"
    )
    print(
        f"Duplicates removed: "
        f"{manifest['stats']['duplicate_id_or_image_path']}"
    )

    for country in manifest["countries"]:
        matched = manifest["path_match_counts"].get(country, 0)
        unique = manifest["eligible_unique_counts"].get(country, 0)
        selected = manifest["selected_counts"].get(country, 0)
        print(
            f"{country:25s} "
            f"path-match={matched:8d} "
            f"unique={unique:8d} "
            f"selected={selected:5d}"
        )

    print("=" * 88)
    print(
        f"Saved selection: "
        f"{args.output_dir / 'selected' / '_all_selected.json'}"
    )
    print(
        f"Manifest: "
        f"{args.output_dir / 'selected' / '_selection_manifest.json'}"
    )

    if args.selection_only:
        print("\n--selection-only requested: stopping before caption generation.")
        return

    caption_selected_samples(
        selected_by_country=selected_by_country,
        args=args,
    )


if __name__ == "__main__":
    main()
