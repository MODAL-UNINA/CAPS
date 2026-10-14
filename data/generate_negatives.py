from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from PIL import Image
from tqdm import tqdm

from utils.utils import (
    atomic_write_json,
    normalize_image_path,
    load_rgb_image,
    build_negative_caption_prompt,
    clean_caption,
)
from utils.config import NEGATIVE_OUTPUT_DIR, CAPTION_MODEL, OLLAMA_BASE_URL, MAX_IMAGE_SIDE, TRAIN_SPLIT, VAL_SPLIT


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if 0 <= start < end:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    return None


def image_to_base64(
    image: Image.Image,
) -> str:
    """
    Encode a PIL image as JPEG base64 for ChatOllama multimodal input.
    """
    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


class VLMNegativeGenerator:
    """
    Local multimodal generator using LangChain ChatOllama.

    ChatOllama supports multimodal messages with an image_url block
    containing an inline base64 data URL.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float = 0.0
    ) -> None:
        self.model_name = model_name

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    def generate(
        self,
        image: Image.Image,
        positive_caption: str,
    ) -> dict[str, Any]:
        prompt = build_negative_caption_prompt(
            positive_caption=positive_caption,
        )

        image_b64 = image_to_base64(
            image
        )

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": (
                        f"data:image/jpeg;base64,"
                        f"{image_b64}"
                    ),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        )

        # ChatOllama runs the selected model through the local Ollama server.
        response = self.llm.invoke(
            [message]
        )

        raw_text = str(
            response.content
        ).strip()

        parsed = extract_json_object(
            raw_text
        )

        if parsed is None:
            raise ValueError(
                "Could not parse JSON from ChatOllama output: "
                f"{raw_text!r}"
            )

        negative_caption = clean_caption(
            parsed.get(
                "negative_caption",
                "",
            )
        )

        original_element = clean_caption(
            parsed.get(
                "original_cultural_element",
                "",
            )
        )

        replacement_element = clean_caption(
            parsed.get(
                "replacement_cultural_element",
                "",
            )
        )

        if not negative_caption:
            raise ValueError(
                "ChatOllama returned an empty negative_caption."
            )

        return {
            "negative_caption": negative_caption,
            "original_cultural_element": original_element,
            "replacement_cultural_element": replacement_element,
            "raw_model_output": raw_text,
        }


def discover_jsons(
    input_dir: Path,
) -> list[Path]:
    names = {
        f"{TRAIN_SPLIT}.json",
        f"{VAL_SPLIT}.json",
    }

    paths = [
        p
        for p in input_dir.rglob("*.json")
        if p.name in names
    ]

    return sorted(
        list(dict.fromkeys(paths))
    )


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if not isinstance(value, list):
        raise ValueError(
            f"{path} must contain a JSON list."
        )

    return [
        x for x in value
        if isinstance(x, dict)
    ]


def resolve_caption(
    sample: dict[str, Any],
) -> str:
    return str(
        sample.get(
            "generated_caption",
            sample.get("caption", ""),
        )
    ).strip()


def process_file(
    input_json: Path,
    output_json: Path,
    generator: VLMNegativeGenerator,
    images_dir: Path,
    args: argparse.Namespace,
) -> None:
    """
    Generate exactly one hard negative for each positive caption.

    No cultural term is supplied to the generator and no quality filters
    are applied. Existing rows are reused so the script can resume.
    """
    samples = load_json(input_json)

    existing: list[dict[str, Any]] = []
    if output_json.exists():
        try:
            existing = load_json(output_json)
        except Exception:
            existing = []

    completed = {
        int(row["sample_index"])
        for row in existing
        if "sample_index" in row
    }

    candidates: list[tuple[int, dict[str, Any], str]] = []

    for sample_idx, sample in enumerate(samples):
        if sample_idx in completed:
            continue

        positive_caption = resolve_caption(sample)
        if not positive_caption:
            continue

        candidates.append(
            (
                sample_idx,
                sample,
                positive_caption,
            )
        )

        if (
            args.max_samples is not None
            and len(candidates) >= args.max_samples
        ):
            break

    results = list(existing)

    print(
        f"\n[{input_json.name}] "
        f"samples={len(samples)}, "
        f"to_generate={len(candidates)}"
    )

    generated_count = 0
    error_count = 0

    try:
        for (
            sample_idx,
            sample,
            positive_caption,
        ) in tqdm(
            candidates,
            desc=input_json.name,
        ):
            image_path = (
                images_dir
                / normalize_image_path(
                    str(sample.get("image", ""))
                )
            )

            try:
                image = load_rgb_image(
                    image_path,
                    max_side=args.max_image_side,
                )
            except Exception as exc:
                results.append(
                    {
                        **sample,
                        "sample_index": sample_idx,
                        "positive_caption": positive_caption,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                error_count += 1
                continue

            try:
                generated = generator.generate(
                    image=image,
                    positive_caption=positive_caption,
                )

                results.append(
                    {
                        **sample,
                        "sample_index": sample_idx,
                        "positive_caption": positive_caption,
                        "negative_caption": generated[
                            "negative_caption"
                        ],
                        "original_cultural_element": generated[
                            "original_cultural_element"
                        ],
                        "replacement_cultural_element": generated[
                            "replacement_cultural_element"
                        ],
                        "raw_model_output": generated[
                            "raw_model_output"
                        ],
                        "status": "generated",
                    }
                )
                generated_count += 1

            except Exception as exc:
                results.append(
                    {
                        **sample,
                        "sample_index": sample_idx,
                        "positive_caption": positive_caption,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                error_count += 1

            # Periodic checkpoint.
            if (
                len(results) % args.save_every == 0
            ):
                atomic_write_json(
                    output_json,
                    results,
                )

    except KeyboardInterrupt:
        print(
            "\nKeyboardInterrupt detected. "
            "Saving progress before exiting..."
        )
        atomic_write_json(
            output_json,
            results,
        )
        print(
            f"Partial results saved to {output_json}"
        )
        raise

    # Final checkpoint.
    atomic_write_json(
        output_json,
        results,
    )

    print(
        f"[{input_json.name}] "
        f"generated={generated_count}, "
        f"errors={error_count}"
    )
    print(
        f"[{input_json.name}] output={output_json}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cultural counterfactual hard-negative captions "
            "with a local ChatOllama multimodal model."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input_json",
        type=Path,
        help="One JSON file to process.",
    )
    source.add_argument(
        "--input_dir",
        type=Path,
        help=(
            "Directory searched recursively for train/val JSON files."
        ),
    )

    parser.add_argument(
        "--images_dir",
        type=Path,
        help="Root directory used to resolve image paths.",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output JSON for --input_json.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=NEGATIVE_OUTPUT_DIR,
        help="Output root for --input_dir.",
    )

    parser.add_argument(
        "--ollama_model",
        default=CAPTION_MODEL,
        help=(
            "Local multimodal model available in Ollama."
        ),
    )

    parser.add_argument(
        "--ollama_base_url",
        default=OLLAMA_BASE_URL,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max_image_side",
        type=int,
        default=MAX_IMAGE_SIDE,
    )


    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of captions to process."
        ),
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"ChatOllama model: {args.ollama_model}"
    )
    print(
        f"Ollama base URL: {args.ollama_base_url}"
    )

    generator = VLMNegativeGenerator(
        model_name=args.ollama_model,
        base_url=args.ollama_base_url,
        temperature=args.temperature,
    )

    if args.input_json is not None:
        output_json = args.output_json

        if output_json is None:
            output_json = (
                args.input_json.parent
                / f"{args.input_json.stem}_negatives.json"
            )

        process_file(
            input_json=args.input_json,
            output_json=output_json,
            generator=generator,
            images_dir=args.images_dir,
            args=args,
        )
        return

    input_files = discover_jsons(
        args.input_dir
    )

    if not input_files:
        raise FileNotFoundError(
            f"No train/val JSON files found under "
            f"{args.input_dir}"
        )

    print(
        f"Found {len(input_files)} JSON files."
    )

    for input_json in input_files:
        relative = input_json.relative_to(
            args.input_dir
        )

        output_json = (
            args.output_dir / relative
        ).with_name(
            f"{relative.stem}_negatives.json"
        )

        process_file(
            input_json=input_json,
            output_json=output_json,
            generator=generator,
            images_dir=args.images_dir,
            args=args,
        )


if __name__ == "__main__":
    main()
