from __future__ import annotations

import argparse
import difflib
import json
import math
import re

from collections import defaultdict
from pathlib import Path
from utils.utils import (
    make_collate_fn,
    set_seed,
    build_dataloaders,
    find_subsequence,
)
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from CAPS.model import ImageConditionedPromptLlava

from utils.config import LEARNING_RATE, EARLY_STOPPING_MIN_DELTA, PROMPT_BOTTLENECK_SIZE
from utils.config import (
    DEFAULT_MODEL,
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    CAPS_OUTPUT_DIR,
    ASSISTANT_MARKER,
)


# ---------------------------------------------------------------------------
# Text alignment utilities
# ---------------------------------------------------------------------------

def tokenize_with_char_spans(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text into word/punctuation units while keeping char offsets."""
    _TOKEN_RE = re.compile(r"\w+(?:[-'][\w]+)*|[^\w\s]")

    return [
        (match.group(0), match.start(), match.end())
        for match in _TOKEN_RE.finditer(text)
    ]


def extract_single_replacement_span(
    positive: str,
    negative: str,
) -> dict[str, Any] | None:
    """
    Find one and only one word-level replacement between two captions.

    We intentionally accept only a single `replace` opcode. This supports
    arbitrary-length spans (e.g. "checkered shirt" -> "white kurta") while
    excluding examples with multiple independent edits, pure insertions, or
    pure deletions.

    Returns character offsets in the ORIGINAL caption strings.
    """
    pos_tokens = tokenize_with_char_spans(positive)
    neg_tokens = tokenize_with_char_spans(negative)

    matcher = difflib.SequenceMatcher(
        a=[token for token, _, _ in pos_tokens],
        b=[token for token, _, _ in neg_tokens],
        autojunk=False,
    )

    non_equal = [
        opcode
        for opcode in matcher.get_opcodes()
        if opcode[0] != "equal"
    ]

    if len(non_equal) != 1:
        return None

    tag, i1, i2, j1, j2 = non_equal[0]
    if tag != "replace":
        return None

    if i1 >= i2 or j1 >= j2:
        return None

    pos_start = pos_tokens[i1][1]
    pos_end = pos_tokens[i2 - 1][2]
    neg_start = neg_tokens[j1][1]
    neg_end = neg_tokens[j2 - 1][2]

    return {
        "positive_span": positive[pos_start:pos_end],
        "negative_span": negative[neg_start:neg_end],
        "positive_char_start": pos_start,
        "positive_char_end": pos_end,
        "negative_char_start": neg_start,
        "negative_char_end": neg_end,
        "positive_token_start": i1,
        "positive_token_end": i2,
        "negative_token_start": j1,
        "negative_token_end": j2,
    }


def build_caption_token_span_mask(
    tokenizer: Any,
    full_input_ids: list[int],
    assistant_ids: list[int],
    caption: str,
    char_start: int,
    char_end: int,
) -> tuple[list[bool], bool]:
    """
    Map a character span in a caption to positions in the full LLaVA token
    sequence.

    The full training text is constructed as:

        USER: <image>\nPROMPT ASSISTANT: CAPTION

    so we tokenize " " + caption to reproduce the leading whitespace that is
    part of the actual full prompt.
    """
    mask = [False] * len(full_input_ids)

    boundary = find_subsequence(
        full_input_ids,
        assistant_ids,
        start=0,
    )
    if boundary < 0:
        return mask, False

    caption_start_in_full = boundary + len(assistant_ids)

    caption_text = " " + caption

    try:
        encoded = tokenizer(
            caption_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return mask, False

    caption_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")

    if caption_ids is None or offsets is None:
        return mask, False

    if caption_ids and isinstance(caption_ids[0], list):
        caption_ids = caption_ids[0]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]

    caption_ids = [int(token_id) for token_id in caption_ids]

    start_idx = find_subsequence(
        full_input_ids,
        caption_ids,
        start=caption_start_in_full,
    )

    if start_idx < 0:
        caption_ids_no_space = tokenizer.encode(
            caption,
            add_special_tokens=False,
        )
        start_idx = find_subsequence(
            full_input_ids,
            [int(x) for x in caption_ids_no_space],
            start=caption_start_in_full,
        )

        if start_idx < 0:
            return mask, False

        # Re-tokenize without leading space and obtain offsets in that string.
        try:
            encoded_no_space = tokenizer(
                caption,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            caption_ids = encoded_no_space["input_ids"]
            offsets = encoded_no_space["offset_mapping"]
            if caption_ids and isinstance(caption_ids[0], list):
                caption_ids = caption_ids[0]
            if offsets and isinstance(offsets[0], list):
                offsets = offsets[0]
        except Exception:
            return mask, False

        offset_shift = 0
    else:
        # Character offsets are relative to " " + caption.
        offset_shift = 1

    span_start = char_start + offset_shift
    span_end = char_end + offset_shift

    for token_offset, (offset_start, offset_end) in enumerate(offsets):
        if offset_end <= offset_start:
            continue
        if offset_end <= span_start:
            continue
        if offset_start >= span_end:
            continue

        full_position = start_idx + token_offset
        if 0 <= full_position < len(mask):
            mask[full_position] = True

    return mask, any(mask)


def make_cultural_span_collate_fn(
    processor: Any,
):
    """
    Reuse the existing dataset collator and add span masks computed entirely
    from positive/negative caption text.
    """
    base_collate = make_collate_fn(
        processor=processor,
        preference=True,
    )

    tokenizer = processor.tokenizer
    assistant_ids = tokenizer.encode(
        ASSISTANT_MARKER,
        add_special_tokens=False,
    )

    def collate(batch: list[dict[str, Any] | None]) -> dict[str, Any]:
        collated = base_collate(batch)
        if not collated:
            return {}

        positive_masks: list[torch.Tensor] = []
        negative_masks: list[torch.Tensor] = []
        valid_flags: list[bool] = []

        positive_input_ids = collated["positive_input_ids"]
        negative_input_ids = collated["negative_input_ids"]

        positive_captions = collated["reference_caption"]
        negative_captions = collated["negative_reference_caption"]

        for row_idx, (positive, negative) in enumerate(
            zip(positive_captions, negative_captions)
        ):
            diff = extract_single_replacement_span(
                positive,
                negative,
            )

            pos_mask = [
                False
            ] * positive_input_ids.size(1)
            neg_mask = [
                False
            ] * negative_input_ids.size(1)

            valid = False

            if diff is not None:
                pos_mask, pos_ok = build_caption_token_span_mask(
                    tokenizer=tokenizer,
                    full_input_ids=positive_input_ids[row_idx].tolist(),
                    assistant_ids=assistant_ids,
                    caption=positive,
                    char_start=diff["positive_char_start"],
                    char_end=diff["positive_char_end"],
                )

                neg_mask, neg_ok = build_caption_token_span_mask(
                    tokenizer=tokenizer,
                    full_input_ids=negative_input_ids[row_idx].tolist(),
                    assistant_ids=assistant_ids,
                    caption=negative,
                    char_start=diff["negative_char_start"],
                    char_end=diff["negative_char_end"],
                )

                valid = bool(pos_ok and neg_ok)

            positive_masks.append(torch.tensor(pos_mask, dtype=torch.bool))
            negative_masks.append(torch.tensor(neg_mask, dtype=torch.bool))
            valid_flags.append(valid)

        collated["positive_span_mask"] = torch.stack(positive_masks, dim=0)
        collated["negative_span_mask"] = torch.stack(negative_masks, dim=0)
        collated["span_valid"] = torch.tensor(
            valid_flags,
            dtype=torch.bool,
        )

        return collated

    return collate


class ImageConditionedPromptLlavaCulturalSpan(ImageConditionedPromptLlava):
    """
    Image-conditioned prompt model with localized cultural-span preference.

    The image features and image-conditioned prompt are computed exactly once
    per image and reused by the positive and negative caption branches.
    """


    @staticmethod
    def _span_scores(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        span_mask: torch.Tensor,
        prompt_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean token log-probability restricted to the cultural span.

        Returns:
            scores: [batch]
            valid: [batch] bool, indicating whether at least one span token
                   was found after causal-LM shifting.
        """

        start = int(prompt_length)
        end = -1

        shifted_logits = logits[:, start:end, :]
        shifted_labels = input_ids[:, 1:].to(shifted_logits.device)
        shifted_span_mask = span_mask[:, 1:].to(shifted_logits.device).bool()

        if shifted_logits.size(1) != shifted_span_mask.size(1):
            raise RuntimeError(
                "Span/logit alignment mismatch after accounting for the "
                f"soft prompt: logits={shifted_logits.shape}, "
                f"span_mask={shifted_span_mask.shape}, "
                f"prompt_length={prompt_length}."
            )

        lengths = shifted_span_mask.sum(dim=1)
        valid = lengths > 0

        # Select only the positions belonging to the cultural span first.
        selected_logits = shifted_logits[shifted_span_mask].float()
        selected_labels = shifted_labels[shifted_span_mask]

        if selected_logits.numel() == 0:
            scores = torch.zeros(
                logits.size(0),
                device=logits.device,
                dtype=torch.float32,
            )
            return scores, valid

        target_logits = selected_logits.gather(
            dim=-1,
            index=selected_labels.unsqueeze(-1),
        ).squeeze(-1)

        token_logprobs = (
            target_logits
            - torch.logsumexp(selected_logits, dim=-1)
        )

        # Map each selected token back to its batch example.
        batch_indices = (
            torch.arange(
                logits.size(0),
                device=logits.device,
            )
            .unsqueeze(1)
            .expand_as(shifted_span_mask)[shifted_span_mask]
        )

        score_sums = torch.zeros(
            logits.size(0),
            device=logits.device,
            dtype=token_logprobs.dtype,
        )
        score_sums.scatter_add_(
            0,
            batch_indices,
            token_logprobs,
        )

        scores = score_sums / lengths.clamp_min(1)
        return scores, valid

    def _prepare_sequence_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        image_features_language: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Build LLaVA inputs using already computed image features/prompt."""
        language_device = self.backbone_device

        input_ids = input_ids.to(language_device)
        attention_mask = attention_mask.to(language_device)

        token_embeddings = self.backbone.get_input_embeddings()(input_ids)
        token_embeddings = self._replace_image_placeholders(
            input_ids=input_ids,
            token_embeddings=token_embeddings,
            image_features=image_features_language,
        )

        extended_embeddings = torch.cat(
            [prompt_embeddings, token_embeddings],
            dim=1,
        )

        prompt_mask = torch.ones(
            input_ids.size(0),
            self.prompt_length,
            dtype=attention_mask.dtype,
            device=language_device,
        )
        extended_attention_mask = torch.cat(
            [prompt_mask, attention_mask],
            dim=1,
        )

        extended_labels = None
        if labels is not None:
            labels = labels.to(language_device)
            prompt_labels = torch.full(
                (input_ids.size(0), self.prompt_length),
                -100,
                dtype=labels.dtype,
                device=language_device,
            )
            extended_labels = torch.cat(
                [prompt_labels, labels],
                dim=1,
            )

        return (
            extended_embeddings,
            extended_attention_mask,
            extended_labels,
        )

    def _lm_forward_from_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        image_features_language: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> Any:
        (
            inputs_embeds,
            extended_mask,
            extended_labels,
        ) = self._prepare_sequence_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_features_language=image_features_language,
            prompt_embeddings=prompt_embeddings,
        )

        return self.backbone.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=extended_labels,
            use_cache=False,
            return_dict=True,
        )

    def _build_shared_prompt(
        self,
        image_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build P(I) once and share it across positive/negative branches."""
        language_device = self.backbone_device

        image_features_language = image_features.to(language_device)

        prompt_input = image_features_language.to(
            device=self.prompt_device,
            dtype=self.prompt_module.base_prompt.dtype,
        )

        prompt_embeddings = self.prompt_module(prompt_input).to(
            device=language_device,
            dtype=self.backbone.get_input_embeddings().weight.dtype,
        )

        return image_features_language, prompt_embeddings

    def forward(
        self,
        positive_pixel_values: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        positive_labels: torch.Tensor,
        negative_pixel_values: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_attention_mask: torch.Tensor,
        negative_labels: torch.Tensor,
        positive_span_mask: torch.Tensor,
        negative_span_mask: torch.Tensor,
        span_valid: torch.Tensor,
        cultural_span_weight: float = 1.0,
        preference_beta: float = 1.0,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        # Same image => one vision forward pass.
        del negative_pixel_values

        image_features = self._encode_images(positive_pixel_values)
        (
            image_features_language,
            prompt_embeddings,
        ) = self._build_shared_prompt(image_features)

        pos_outputs = self._lm_forward_from_features(
            input_ids=positive_input_ids,
            attention_mask=positive_attention_mask,
            labels=positive_labels,
            image_features_language=image_features_language,
            prompt_embeddings=prompt_embeddings,
        )

        if cultural_span_weight <= 0.0:
            return {
                "loss": pos_outputs.loss,
                "caption_loss": pos_outputs.loss,
                "span_preference_loss": torch.tensor(0.0),
                "span_preference_margin": torch.tensor(0.0),
                "span_preference_accuracy": torch.tensor(0.0),
                "span_valid_fraction": torch.tensor(0.0),
                "span_preference_correct": torch.zeros_like(
                    positive_input_ids,
                    dtype=torch.float32,
                ),
            }

        neg_outputs = self._lm_forward_from_features(
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
            labels=negative_labels,
            image_features_language=image_features_language,
            prompt_embeddings=prompt_embeddings,
        )

        # Localized cultural span scores.
        positive_span_score, pos_span_valid = self._span_scores(
            pos_outputs.logits,
            positive_input_ids,
            positive_span_mask,
            self.prompt_length,
        )
        negative_span_score, neg_span_valid = self._span_scores(
            neg_outputs.logits,
            negative_input_ids,
            negative_span_mask,
            self.prompt_length,
        )

        span_valid = (
            span_valid.to(pos_span_valid.device)
            & pos_span_valid
            & neg_span_valid
        )

        span_margin = positive_span_score - negative_span_score

        if span_valid.any():
            preference_loss = -F.logsigmoid(
                preference_beta * span_margin[span_valid]
            ).mean()
            preference_margin = span_margin[span_valid].mean()
            preference_correct = (
                span_margin > 0
            ).float().detach()
            preference_accuracy = preference_correct[
                span_valid
            ].mean()
        else:
            # Keep a valid autograd scalar while contributing zero gradient.
            preference_loss = pos_outputs.logits.sum() * 0.0
            preference_margin = span_margin.mean() * 0.0
            preference_correct = torch.zeros_like(
                span_margin,
                dtype=torch.float32,
            )
            preference_accuracy = span_margin.mean() * 0.0

        total_loss = (
            pos_outputs.loss
            + cultural_span_weight * preference_loss
        )

        return {
            "loss": total_loss,
            "caption_loss": pos_outputs.loss,
            "span_preference_loss": preference_loss,
            "span_preference_margin": preference_margin,
            "span_preference_accuracy": preference_accuracy,
            "span_valid_fraction": span_valid.float().mean(),
            "span_preference_correct": preference_correct,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: ImageConditionedPromptLlavaCulturalSpan,
    loader: DataLoader,
    cultural_span_weight: float,
    preference_beta: float,
) -> dict[str, Any]:
    model.eval()

    sums: dict[str, float] = defaultdict(float)
    count_batches = 0
    count_examples = 0

    for batch in loader:
        if not batch:
            continue

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            outputs = model(
                positive_pixel_values=batch["positive_pixel_values"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                positive_labels=batch["positive_labels"],
                negative_pixel_values=batch["negative_pixel_values"],
                negative_input_ids=batch["negative_input_ids"],
                negative_attention_mask=batch["negative_attention_mask"],
                negative_labels=batch["negative_labels"],
                positive_span_mask=batch["positive_span_mask"],
                negative_span_mask=batch["negative_span_mask"],
                span_valid=batch["span_valid"],
                cultural_span_weight=cultural_span_weight,
                preference_beta=preference_beta,
            )

        bsz = batch["positive_input_ids"].size(0)
        count_batches += 1
        count_examples += bsz

        for key in (
            "loss",
            "caption_loss",
            "span_preference_loss",
            "span_preference_margin",
            "span_preference_accuracy",
            "span_valid_fraction",
        ):
            sums[key] += float(outputs[key].item())

    denom = max(count_batches, 1)

    result: dict[str, Any] = {
        key: sums[key] / denom
        for key in (
            "loss",
            "caption_loss",
            "span_preference_loss",
            "span_preference_margin",
            "span_preference_accuracy",
            "span_valid_fraction",
        )
    }

    result["perplexity"] = math.exp(
        min(float(result["caption_loss"]), 20.0)
    )

    result["examples"] = count_examples

    return result


def save_checkpoint(
    model: ImageConditionedPromptLlavaCulturalSpan,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    tag: str,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    early_stop_best_loss: float,
    epochs_without_improvement: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = output_dir / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_prompt_module(
        checkpoint_dir / "image_conditioned_prompt.pt"
    )

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "early_stop_best_loss": early_stop_best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
        },
        checkpoint_dir / "trainer_state.pt",
    )

    model.processor.save_pretrained(checkpoint_dir / "processor")
    print(f"Saved checkpoint: {checkpoint_dir}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if args.early_stopping_start_epoch < 1:
        raise ValueError("early_stopping_start_epoch must be >= 1.")
    if args.cultural_span_weight < 0:
        raise ValueError("cultural_span_weight must be >= 0.")
    if args.preference_beta <= 0:
        raise ValueError("preference_beta must be > 0.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This training script requires a CUDA GPU for LLaVA-1.5 7B."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ImageConditionedPromptLlavaCulturalSpan(
        model_name=args.model_name,
        prompt_length=args.prompt_length,
        bottleneck_size=args.bottleneck_size,
        prompt_dropout=args.prompt_dropout,
        residual_scale=args.residual_scale,
        load_in_4bit=args.load_in_4bit,
        compute_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    
    model.prompt_module.to(
        device=model.backbone_device,
        dtype=torch.float32,
    )

    if args.init_prompt_text:
        model.initialize_base_prompt_from_text(
            args.init_prompt_text
        )
        print(
            "Initialized base prompt from text: "
            f"{args.init_prompt_text!r}"
        )

    if args.init_from_static_prompt:
        model.initialize_base_prompt_from_static_checkpoint(
            args.init_from_static_prompt
        )
        print(
            "Initialized base prompt from static checkpoint: "
            f"{args.init_from_static_prompt}"
        )

    if args.resume_prompt_module:
        model.load_prompt_module(args.resume_prompt_module)
        print(
            "Loaded image-conditioned prompt module from "
            f"{args.resume_prompt_module}"
        )

    train_loader, val_loader = build_dataloaders(model.processor, args, preference=True,
        collate_fn=make_cultural_span_collate_fn(model.processor))

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.prompt_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / args.grad_accum
    )

    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    early_stop_best_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume_trainer_state:
        state = torch.load(
            args.resume_trainer_state,
            map_location="cpu",
        )

        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val_loss = float(
            state.get("best_val_loss", float("inf"))
        )
        early_stop_best_loss = float(
            state.get(
                "early_stop_best_loss",
                best_val_loss,
            )
        )
        epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )

        print(
            "Resumed trainer state from "
            f"{args.resume_trainer_state}"
        )
        print(
            "Restored early-stopping state: "
            f"best={early_stop_best_loss:.4f}, "
            f"epochs_without_improvement="
            f"{epochs_without_improvement}"
        )

    history_path = output_dir / "history.json"

    if history_path.exists():
        try:
            history: list[dict[str, Any]] = json.loads(
                history_path.read_text(encoding="utf-8")
            )
        except Exception:
            history = []
    else:
        history = []

    optimizer.zero_grad(set_to_none=True)

    # ==================================================================
    # Epoch loop
    # ==================================================================
    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        model.train()

        running = defaultdict(float)
        running_count = 0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            if not batch:
                continue

            is_last_batch = step == len(train_loader)
            is_update_step = (
                step % args.grad_accum == 0
                or is_last_batch
            )

            remainder = len(train_loader) % args.grad_accum
            in_final_window = (
                remainder != 0
                and step > len(train_loader) - remainder
            )

            divisor = (
                remainder
                if in_final_window
                else args.grad_accum
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                outputs = model(
                    positive_pixel_values=batch[
                        "positive_pixel_values"
                    ],
                    positive_input_ids=batch[
                        "positive_input_ids"
                    ],
                    positive_attention_mask=batch[
                        "positive_attention_mask"
                    ],
                    positive_labels=batch[
                        "positive_labels"
                    ],
                    negative_pixel_values=batch[
                        "negative_pixel_values"
                    ],
                    negative_input_ids=batch[
                        "negative_input_ids"
                    ],
                    negative_attention_mask=batch[
                        "negative_attention_mask"
                    ],
                    negative_labels=batch[
                        "negative_labels"
                    ],
                    positive_span_mask=batch[
                        "positive_span_mask"
                    ],
                    negative_span_mask=batch[
                        "negative_span_mask"
                    ],
                    span_valid=batch[
                        "span_valid"
                    ],
                    cultural_span_weight=(
                        args.cultural_span_weight
                    ),
                    preference_beta=args.preference_beta,
                )

            loss = outputs["loss"]
            (loss / divisor).backward()

            running["loss"] += float(
                outputs["loss"].item()
            )
            running["caption_loss"] += float(
                outputs["caption_loss"].item()
            )
            running["span_preference_loss"] += float(
                outputs["span_preference_loss"].item()
            )
            running["span_margin"] += float(
                outputs["span_preference_margin"].item()
            )
            running["span_accuracy"] += float(
                outputs["span_preference_accuracy"].item()
            )
            running["span_valid_fraction"] += float(
                outputs["span_valid_fraction"].item()
            )
            running_count += 1

            if not is_update_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                model.prompt_module.parameters(),
                args.max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every_updates == 0:
                denom = max(running_count, 1)

                payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "phase": "train",
                    "loss": running["loss"] / denom,
                    "caption_loss": (
                        running["caption_loss"] / denom
                    ),
                    "span_preference_loss": (
                        running["span_preference_loss"] / denom
                    ),
                    "span_preference_margin": (
                        running["span_margin"] / denom
                    ),
                    "span_preference_accuracy": (
                        running["span_accuracy"] / denom
                    ),
                    "span_valid_fraction": (
                        running["span_valid_fraction"] / denom
                    ),
                    "lr": scheduler.get_last_lr()[0],
                }

                print(
                    f"epoch={epoch} "
                    f"update={global_step}/{total_updates} "
                    f"loss={payload['loss']:.4f} "
                    f"cap={payload['caption_loss']:.4f} "
                    f"span={payload['span_preference_loss']:.4f} "
                    f"margin="
                    f"{payload['span_preference_margin']:.4f} "
                    f"SPAN-Acc="
                    f"{payload['span_preference_accuracy']:.3f} "
                    f"SPAN-Cov="
                    f"{payload['span_valid_fraction']:.3f}"
                )

                history.append(payload)
                history_path.write_text(
                    json.dumps(
                        history,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                running = defaultdict(float)
                running_count = 0

            if (
                args.eval_every_updates > 0
                and global_step % args.eval_every_updates == 0
            ):
                val = evaluate(
                    model=model,
                    loader=val_loader,
                    cultural_span_weight=args.cultural_span_weight,
                    preference_beta=args.preference_beta,
                )

                print(
                    f"validation update={global_step}: "
                    f"loss={val['loss']:.4f} "
                    f"cap={val['caption_loss']:.4f} "
                    f"span={val['span_preference_loss']:.4f} "
                    f"margin="
                    f"{val['span_preference_margin']:.4f} "
                    f"SPAN-Acc="
                    f"{val['span_preference_accuracy']:.3f} "
                    f"SPAN-Cov="
                    f"{val['span_valid_fraction']:.3f}"
                )

                history.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "phase": "val_update",
                        **val,
                    }
                )

                history_path.write_text(
                    json.dumps(
                        history,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                model.train()

        # --------------------------------------------------------------
        # End-of-epoch validation
        # --------------------------------------------------------------
        val = evaluate(
            model=model,
            loader=val_loader,
            cultural_span_weight=args.cultural_span_weight,
            preference_beta=args.preference_beta,
        )

        current_val_loss = float(val["loss"])

        print(
            f"end epoch {epoch}: "
            f"val_loss={current_val_loss:.4f} "
            f"val_cap={val['caption_loss']:.4f} "
            f"val_span={val['span_preference_loss']:.4f} "
            f"val_margin="
            f"{val['span_preference_margin']:.4f} "
            f"val_SPAN-Acc="
            f"{val['span_preference_accuracy']:.3f} "
            f"val_SPAN-Cov="
            f"{val['span_valid_fraction']:.3f}"
        )

        improved_checkpoint = (
            current_val_loss < best_val_loss
        )

        if improved_checkpoint:
            best_val_loss = current_val_loss

        early_stopping_enabled = (
            args.early_stopping_patience > 0
        )

        improved_for_early_stopping = (
            current_val_loss
            < early_stop_best_loss
            - args.early_stopping_min_delta
        )

        if improved_for_early_stopping:
            early_stop_best_loss = current_val_loss
            epochs_without_improvement = 0

            print(
                "Early stopping: validation loss improved to "
                f"{early_stop_best_loss:.4f}."
            )
        elif (
            early_stopping_enabled
            and epoch >= args.early_stopping_start_epoch
        ):
            epochs_without_improvement += 1
            print(
                "Early stopping: no improvement larger than "
                f"min_delta={args.early_stopping_min_delta:g} "
                f"({epochs_without_improvement}/"
                f"{args.early_stopping_patience} epochs)."
            )
        elif early_stopping_enabled:
            print(
                "Early stopping: warm-up phase; monitoring "
                "has not started yet."
            )

        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "phase": "val_epoch",
                **val,
                "best_val_loss": best_val_loss,
                "early_stop_best_loss": early_stop_best_loss,
                "epochs_without_improvement": (
                    epochs_without_improvement
                ),
            }
        )

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if improved_checkpoint:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=output_dir,
                tag="best",
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                early_stop_best_loss=early_stop_best_loss,
                epochs_without_improvement=epochs_without_improvement,
                args=args,
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            tag=f"epoch_{epoch}",
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            early_stop_best_loss=early_stop_best_loss,
            epochs_without_improvement=epochs_without_improvement,
            args=args,
        )

        should_stop = (
            early_stopping_enabled
            and epoch >= args.early_stopping_start_epoch
            and epochs_without_improvement
            >= args.early_stopping_patience
        )

        if should_stop:
            print(
                "Early stopping triggered at epoch "
                f"{epoch}. Best monitored validation loss: "
                f"{early_stop_best_loss:.4f}."
            )
            break

    print(
        "Training completed/stopped. "
        f"Best validation loss: {best_val_loss:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-conditioned LLaVA soft-prompt tuning with "
            "localized cultural-span preference loss."
        )
    )

    parser.add_argument("--train_split", default=TRAIN_SPLIT)
    parser.add_argument("--val_split", default=VAL_SPLIT)
    parser.add_argument(
        "--data_dir",
        required=True,
    )
    parser.add_argument(
        "--images_dir",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=str(CAPS_OUTPUT_DIR),
    )

    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--caption_prompt",
        default=DEFAULT_CAPTION_PROMPT,
    )

    parser.add_argument(
        "--init_prompt_text",
        default=None,
        help=(
            "Optional natural-language text used to initialize "
            "the trainable base prompt."
        ),
    )
    parser.add_argument(
        "--init_from_static_prompt",
        default=None,
        help=(
            "Optional static soft-prompt checkpoint. Initializes only "
            "the base prompt of the image-conditioned module."
        ),
    )
    parser.add_argument(
        "--resume_prompt_module",
        default=None,
        help=(
            "Optional image-conditioned prompt module checkpoint "
            "(image_conditioned_prompt.pt)."
        ),
    )
    parser.add_argument(
        "--resume_trainer_state",
        default=None,
        help="Optional trainer_state.pt checkpoint.",
    )

    parser.add_argument(
        "--prompt_length",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--bottleneck_size",
        type=int,
        default=PROMPT_BOTTLENECK_SIZE,
    )
    parser.add_argument(
        "--prompt_dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--cultural_span_weight",
        type=float,
        default=1,
        help=(
            "lambda_span multiplying the localized cultural preference loss."
        ),
    )
    parser.add_argument(
        "--preference_beta",
        type=float,
        default=1.0,
        help="beta multiplying the cultural span preference margin.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help=(
            "Stop after this many end-of-epoch validations without "
            "a sufficient improvement. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
        help=(
            "Minimum validation-loss decrease required to reset "
            "early stopping patience."
        ),
    )
    parser.add_argument(
        "--early_stopping_start_epoch",
        type=int,
        default=1,
    )

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log_every_updates", type=int, default=10)
    parser.add_argument("--eval_every_updates", type=int, default=100)

    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
