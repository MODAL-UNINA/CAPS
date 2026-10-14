"""Shared data, text and model utilities for CAPS training and evaluation."""
import csv
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from utils.config import (
    ASSISTANT_MARKER,
    DEFAULT_CAPTION_PROMPT,
    IMAGE_PREFIX,
    LEGACY_IMAGE_PREFIX,
    MAX_IMAGE_SIDE,
)


def normalize_image_path(raw):
    path = str(raw).strip().replace('\\', '/')
    if path.startswith(LEGACY_IMAGE_PREFIX):
        path = path.replace(LEGACY_IMAGE_PREFIX, IMAGE_PREFIX, 1)
    return path


def load_rgb_image(value, max_side=None):
    if isinstance(value, dict):
        value = io.BytesIO(value['bytes']) if value.get('bytes') else value.get('path')
    if isinstance(value, Image.Image):
        image = value.copy()
    else:
        with Image.open(value) as source:
            source.load()
            image = source.copy()
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if image.mode in ('RGBA', 'LA') or image.mode == 'P' and 'transparency' in image.info:
        rgba = image.convert('RGBA')
        image = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        image.alpha_composite(rgba)
    return image.convert('RGB')


def load_records(path, split=None):
    path = Path(path)
    if path.is_dir():
        from datasets import DatasetDict, load_from_disk
        records = load_from_disk(str(path))
        if isinstance(records, DatasetDict):
            if split is None:
                choices = [key for key in ('validation', 'val', 'test') if key in records]
                if len(choices) != 1:
                    raise ValueError(f'Specify --split; available: {list(records)}')
                split = choices[0]
            records = records[split]
        return records
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    records = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(records, dict):
        records = records.get('samples', records)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f'{path}: expected list of records or object with samples')
    return records


def resolve_image(row, images_dir, field=None):
    for key in ([field] if field else ['image_path', 'image', 'img_fn', 'image_file']):
        value = row.get(key)
        if value is None or isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and value.get('path') and not value.get('bytes'):
            value = value['path']
        if isinstance(value, (str, Path)):
            path = Path(str(value).replace('\\', '/')).expanduser()
            return path if path.is_absolute() else Path(images_dir) / path
        return value
    raise ValueError('No image field found')


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items() if not isinstance(v, (Image.Image, bytes))}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return json_ready(value.tolist())
    return value


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


# Prompt construction

def build_negative_caption_prompt(
    positive_caption: str,
) -> str:
    """
    Ask the VLM to create the cultural counterfactual autonomously.

    No cultural word/term is provided to the generator. The model must inspect
    the image and the positive caption, identify one culturally meaningful
    element, and minimally change that element so that the new caption becomes
    false for the image while remaining otherwise very similar.
    """
    return f"""
You are generating a CULTURAL COUNTERFACTUAL HARD NEGATIVE for an
image-captioning dataset.

You are given:
1. an image;
2. a correct positive caption.

Positive caption:
"{positive_caption}"

Your task:
Create EXACTLY ONE negative caption by minimally modifying the positive
caption.

The negative must:
1. remain very close to the positive in wording, syntax, scene description,
   people, objects, actions, and overall semantics;
2. change ONLY ONE culturally meaningful visual detail;
3. replace it with a culturally plausible alternative that would be
   inconsistent with what is actually visible in the image;
4. preserve caption length and grammatical structure;

The cultural difference can concern, for example, clothing, religious objects,
architecture, food, ritual/practice, symbols, instruments, or other
culturally meaningful visual attributes. Choose ONLY a change that is
plausibly grounded in the image and the positive caption.

Return JSON only:
{{
  "negative_caption": "...",
  "original_cultural_element": "...",
  "replacement_cultural_element": "..."
}}
""".strip()


def format_caption_prompt(instruction=DEFAULT_CAPTION_PROMPT):
    """LLaVA conversation prefix used by every training and inference path."""
    return f"USER: <image>\n{instruction} {ASSISTANT_MARKER}"


def build_positive_caption_prompt(question, answer, country, style_instruction):
    return f"""
Question: {question}
Answer: {answer}
Country metadata: {country}

Requested style:
{style_instruction}

Generate the final caption.
""".strip()


def build_judge_user_prompt(references, caption):
    return '\n\n'.join(f'Reference {i}:\n{text}' for i, text in enumerate(references, 1)) + f'\n\nCandidate:\n{caption}'


# Text normalization and alignment

def find_subsequence(
    sequence: list[int],
    subsequence: list[int],
    start: int = 0,
) -> int:
    """Return first subsequence start at/after `start`, or -1."""
    n = len(subsequence)
    if n == 0:
        return -1

    max_pos = len(sequence) - n
    for pos in range(max(start, 0), max_pos + 1):
        if sequence[pos : pos + n] == subsequence:
            return pos
    return -1


def clean_caption(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(caption|final caption|negative caption|counterfactual caption)\s*:\s*", "", text, flags=re.I)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


# Reproducible data selection

def stable_country_seed(seed: int, country: str) -> int:
    return int.from_bytes(f"{seed}:{country}".encode("utf-8"), "big")


# Cultural vocabulary

def load_cultural_words(path):
    words = {line.strip().lower() for line in Path(path).read_text(encoding='utf-8').splitlines()}
    words.discard('')
    if not words:
        raise ValueError(f'Empty cultural vocabulary: {path}')
    return words


# Dataset and collator

class CulturalCaptioningDataset(Dataset):
    def __init__(self, json_path, images_dir, prompt=DEFAULT_CAPTION_PROMPT, *, preference=False):
        self.images_dir, self.prompt = Path(images_dir), prompt
        records = json.loads(Path(json_path).read_text(encoding='utf-8'))
        if not isinstance(records, list):
            raise ValueError(f'{json_path} must contain a JSON list')
        self.samples = []
        for row in records:
            if not isinstance(row, dict):
                continue
            image = row.get('image')
            caption = row.get('generated_caption') or row.get('caption')
            negative = row.get('negative_caption')
            if not isinstance(image, str) or not image.strip() or not isinstance(caption, str) or not caption.strip():
                continue
            if preference and (not isinstance(negative, str) or not negative.strip()):
                continue
            self.samples.append({'image': normalize_image_path(image), 'caption': caption.strip(),
                                 'negative_caption': negative.strip() if isinstance(negative, str) else ''})
        if len(records) != len(self.samples):
            print(f'[{json_path}] skipped {len(records)-len(self.samples)} invalid records')
        if not self.samples:
            raise ValueError(f'No valid records in {json_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        prefix = format_caption_prompt(self.prompt)
        return {'image': load_rgb_image(self.images_dir / row['image'], max_side=MAX_IMAGE_SIDE),
                'full_text': f"{prefix} {row['caption']}",
                'negative_full_text': f"{prefix} {row['negative_caption']}",
                'caption': row['caption'], 'negative_caption': row['negative_caption']}


def discover_split(data_dir, images_dir, split, prompt=DEFAULT_CAPTION_PROMPT, *, preference=False):
    """Use a root split OR one-level subfolders, avoiding duplicated data."""
    base = Path(data_dir)
    direct = base / f'{split}.json'
    paths = [direct] if direct.is_file() else sorted(base.glob(f'*/{split}.json'))
    if not paths:
        raise FileNotFoundError(f'No {split}.json under {base}')
    datasets = [CulturalCaptioningDataset(path, images_dir, prompt, preference=preference) for path in paths]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def make_collate_fn(processor, *, preference=False):
    tokenizer = processor.tokenizer
    assistant_ids = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
    if not assistant_ids:
        raise ValueError('Empty ASSISTANT marker')
    image_id = getattr(processor, 'image_token_id', None)
    if image_id is None:
        image_id = tokenizer.convert_tokens_to_ids('<image>')

    def encode(batch, key):
        encoded = processor(images=[row['image'] for row in batch],
                            text=[row[key] + (tokenizer.eos_token or '') for row in batch],
                            padding=True, return_tensors='pt')
        ids = encoded['input_ids']
        labels = ids.clone()
        for index, row in enumerate(ids.tolist()):
            start = find_subsequence(row, assistant_ids)
            if start < 0:
                raise ValueError('ASSISTANT boundary missing')
            labels[index, :start+len(assistant_ids)] = -100
            labels[index, encoded['attention_mask'][index] == 0] = -100
            labels[index, ids[index] == image_id] = -100
            if not (labels[index] != -100).any():
                raise ValueError('Caption has no target tokens')
        return {'pixel_values': encoded['pixel_values'], 'input_ids': ids,
                'attention_mask': encoded['attention_mask'], 'labels': labels}

    def collate(batch):
        if not batch:
            return {}
        positive = encode(batch, 'full_text')
        if not preference:
            return positive
        negative = encode(batch, 'negative_full_text')
        return {**{f'positive_{k}': v for k, v in positive.items()},
                **{f'negative_{k}': v for k, v in negative.items()},
                'reference_caption': [row['caption'] for row in batch],
                'negative_reference_caption': [row['negative_caption'] for row in batch]}
    return collate


# Training setup

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(processor, args, *, preference=False, collate_fn=None):
    if collate_fn is None:
        collate_fn = make_collate_fn(processor, preference=preference)
    loaders = []
    for split, batch_size, shuffle in (
        (args.train_split, args.batch_size, True),
        (args.val_split, args.eval_batch_size, False),
    ):
        dataset = discover_split(args.data_dir, args.images_dir, split,
                                 args.caption_prompt, preference=preference)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=args.num_workers, pin_memory=True,
                                  persistent_workers=args.num_workers > 0,
                                  collate_fn=collate_fn, drop_last=False))
    return tuple(loaders)


# Frozen backbone and image embeddings

def load_llava(model_name, load_in_4bit, compute_dtype, device_map):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None
    backbone = AutoModelForImageTextToText.from_pretrained(
        model_name, quantization_config=quantization,
        torch_dtype=compute_dtype, device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.patch_size = backbone.config.vision_config.patch_size
    processor.vision_feature_select_strategy = backbone.config.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return backbone, processor


class FrozenVisionMixin:
    """Stateless operations shared by all three tuning wrappers."""

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the frozen vision tower and multimodal projector once."""
        vision_device = next(self.backbone.vision_tower.parameters()).device
        projector_device = next(
            self.backbone.multi_modal_projector.parameters()
        ).device

        with torch.no_grad():
            pixel_values = pixel_values.to(
                device=vision_device, dtype=next(self.backbone.vision_tower.parameters()).dtype
            )
            vision_outputs = self.backbone.vision_tower(
                pixel_values,
                output_hidden_states=True,
            )
            selected = vision_outputs.hidden_states[
                self.backbone.config.vision_feature_layer
            ]
            if self.backbone.config.vision_feature_select_strategy == "default":
                selected = selected[:, 1:]

            image_features = self.backbone.multi_modal_projector(
                selected.to(projector_device)
            )
        return image_features

    def _replace_image_placeholders(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        image_token_index = self.backbone.config.image_token_index
        mask = input_ids.eq(image_token_index).to(token_embeddings.device)

        expected = image_features.shape[0] * image_features.shape[1]
        actual = int(mask.sum().item())
        if actual != expected:
            raise ValueError(
                f"Image token mismatch: found {actual} placeholder tokens, "
                f"but obtained {expected} image features."
            )

        return token_embeddings.masked_scatter(
            mask.unsqueeze(-1).expand_as(token_embeddings),
            image_features.to(
                device=token_embeddings.device,
                dtype=token_embeddings.dtype,
            ),
        )
