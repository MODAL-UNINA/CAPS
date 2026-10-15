from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from utils.utils import (
    discover_split,
    make_collate_fn,
    set_seed,
)

import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from prompt_tuning.model import StaticPromptLlava

from utils.config import LEARNING_RATE, EARLY_STOPPING_MIN_DELTA
from utils.config import DEFAULT_MODEL, DEFAULT_CAPTION_PROMPT, DEFAULT_SEED, TRAIN_SPLIT, VAL_SPLIT, PROMPT_OUTPUT_DIR


class PromptOnlyTrainer(Trainer):

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        del state_dict  
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_adapter(directory / "adapter")
        self.model.save_prompt_state(directory / "soft_prompt.pt")
        torch.save(self.args, directory / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
        target = model if model is not None else self.model
        path = Path(resume_from_checkpoint) / "soft_prompt.pt"
        if not path.exists():
            raise FileNotFoundError(f"Soft prompt checkpoint not found: {path}")
        target.load_prompt_state(path)

    def _load_best_model(self) -> None:
        if self.state.best_model_checkpoint is not None:
            self.model.load_prompt_state(
                Path(self.state.best_model_checkpoint) / "soft_prompt.pt"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
    )
    parser.add_argument(
        "--images_dir", required=True
    )
    parser.add_argument(
        "--output_dir", default=str(PROMPT_OUTPUT_DIR)
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--caption_prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument(
        "--init_prompt_text",
        default=None,
        help="Initialization text (default: --caption_prompt).",
    )
    parser.add_argument(
        "--random_init", action=argparse.BooleanOptionalAction, default=True,
        help="Use random PEFT initialization.",
    )
    parser.add_argument(
        "--prompt_position", choices=["after_bos", "prepend"],
        default="prepend",
    )
    parser.add_argument(
        "--debug_first_batch", action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare baseline LLaVA, manual embedding, and PEFT losses at startup.",
    )
    parser.add_argument(
        "--debug_only", action="store_true", default=False,
        help="Run diagnostics only, without starting training.",
    )
    parser.add_argument("--prompt_length", type=int, default=16)
    parser.add_argument("--train_split", default=TRAIN_SPLIT)
    parser.add_argument("--val_split", default=VAL_SPLIT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log_every_updates", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=EARLY_STOPPING_MIN_DELTA)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume_from_checkpoint", default=None,
        help="Path to a complete checkpoint-XXX directory (adapter and Trainer state)",
    )
    return parser.parse_args()


def _preflight_loss_check(
    model: StaticPromptLlava,
    train_set,
    collate_fn,
) -> None:
    """Compare losses on the same example, with identical masks, weights, and image.

    1) Native multimodal LLaVA forward pass without virtual tokens.
    2) The same language model with reconstructed multimodal embeddings and no virtual tokens.
    3) The same language model with reconstructed embeddings and a PEFT prompt.

    If 1 differs from 2, inspect visual feature insertion and loss computation.
    If 1 is close to 2 but 3 is much worse, inspect the soft prompt and its position.
    """
    tokenizer = model.processor.tokenizer
    model.eval()

    device = model.language_device
    tower_param = next(model.backbone.vision_tower.parameters(), None)
    if tower_param is None:
        raise RuntimeError("Vision tower has no parameters before the forward pass")
    batch = {k: value.to(device) for k, value in batch.items()}
    pixel_values = batch["pixel_values"].to(
        device=tower_param.device, dtype=tower_param.dtype
    )
    image_embeddings = model.backbone.get_input_embeddings().weight
    prompt_embeddings = model.peft_lm.prompt_encoder[
        model.peft_lm.active_adapter
    ].embedding.weight

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Native LLaVA forward pass with the same tokenizer and labels.
        baseline = model.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=pixel_values,
            labels=batch["labels"],
            use_cache=False,
            return_dict=True,
        )
        native_loss = baseline.loss.float().item()
        del baseline

        # Isolate differences caused by manual multimodal embedding construction.
        multimodal = model._multimodal_embeds(
            pixel_values, batch["input_ids"]
        )
        no_prompt = model.backbone.language_model(
            inputs_embeds=multimodal,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
            return_dict=True,
        )
        manual_loss = no_prompt.loss.float().item()
        del no_prompt, multimodal

        with_prompt = model(**batch)
        prompt_loss = with_prompt["loss"].float().item()
        del with_prompt

    print(f"Baseline LLaVA loss:            {native_loss:.4f}")
    print(f"Manual embedding loss (no PEFT):  {manual_loss:.4f}")
    print(f"PEFT prompt loss:              {prompt_loss:.4f}")
    if not all(math.isfinite(v) for v in (native_loss, manual_loss, prompt_loss)):
        raise FloatingPointError("Non-finite loss: check images, labels, and quantization")
    if abs(native_loss - manual_loss) > 0.25:
        print(
            "[WARNING] Native LLaVA and manual reconstruction differ: "
            "check visual features and loss computation."
        )
    if prompt_loss - manual_loss > 3.0:
        print(
            "[WARNING] The soft prompt substantially increases the initial loss: "
            "check prompt initialization, position, and embeddings."
        )
    if native_loss > 10:
        print(
            "[WARNING] LLaVA has a high loss even without the soft prompt: "
            "check generated_caption, images, and the label mask."
        )
    print("==========================================\n", flush=True)
    model.train()

def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU for LLaVA-1.5 7B")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The bf16 configuration requires a CUDA GPU with bf16 support")
    if args.prompt_length < 1:
        raise ValueError("prompt_length must be >= 1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This quantized wrapper does not support torch.nn.DataParallel. "
            "Run: CUDA_VISIBLE_DEVICES=0 python -m prompt_tuning.train [options]"
        )
    if args.debug_only and not args.debug_first_batch:
        raise ValueError("--debug_only requires --debug_first_batch")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model = StaticPromptLlava(
        model_name=args.model_name,
        prompt_length=args.prompt_length,
        load_in_4bit=args.load_in_4bit,
        init_prompt_text=(
            None if args.random_init else (args.init_prompt_text or args.caption_prompt)
        ),
        prompt_position=args.prompt_position,
        compute_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

            
    train_set = discover_split(
        args.data_dir, args.images_dir, args.train_split, args.caption_prompt
    )
    val_set = discover_split(
        args.data_dir, args.images_dir, args.val_split, args.caption_prompt
    )
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    collate_fn = make_collate_fn(model.processor)
    if args.debug_first_batch:
        _preflight_loss_check(model, train_set, collate_fn)
    if args.debug_only:
        print("Diagnostics complete; training not started (--debug_only).")
        return

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.log_every_updates,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        prediction_loss_only=True,
        label_names=["labels"],
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_min_delta,
            )
        )

    trainer = PromptOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collate_fn,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    print("Validation:", metrics)

    # Trainer has now reloaded the best soft prompt in place.
    best_dir = output_dir / "best_adapter"
    model.save_adapter(best_dir)
    model.processor.save_pretrained(best_dir / "processor")
    (best_dir / "base_model.txt").write_text(args.model_name, encoding="utf-8")
    print(f"Best PEFT soft prompt saved to: {best_dir}")


if __name__ == "__main__":
    main()
