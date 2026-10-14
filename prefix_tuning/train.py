from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from prefix_tuning.model import PrefixTuningLlava

from utils.config import LEARNING_RATE, EARLY_STOPPING_MIN_DELTA
from utils.config import DEFAULT_MODEL, DEFAULT_CAPTION_PROMPT, DEFAULT_SEED, TRAIN_SPLIT, VAL_SPLIT, PREFIX_OUTPUT_DIR
from utils.utils import set_seed, build_dataloaders


@torch.no_grad()
def evaluate(
    model: PrefixTuningLlava,
    loader: DataLoader,
    device_type: str,
) -> dict[str, object]:
    model.eval()

    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        if not batch:
            continue

        with torch.amp.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        loss_value = float(outputs["loss"].item())

        total_loss += loss_value
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)

    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),

    }


def save_checkpoint(
    model: PrefixTuningLlava,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: Path,
    tag: str,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = output_dir / tag
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_prefix_adapter(checkpoint_dir)

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        checkpoint_dir / "trainer_state.pt",
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Prefix-tuning this LLaVA-1.5 7B setup requires CUDA."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (output_dir / "args.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False,
        )

    model = PrefixTuningLlava(
        model_name=args.model_name,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=args.prefix_projection,
        prefix_projection_hidden_size=args.prefix_projection_hidden_size,
        load_in_4bit=args.load_in_4bit,
        compute_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    train_loader, val_loader = build_dataloaders(model.processor, args)

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable Prefix-Tuning parameters were found."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(
        len(train_loader)
        / max(args.grad_accum, 1)
    )
    total_updates = max(
        steps_per_epoch * args.epochs,
        1,
    )
    warmup_steps = int(
        total_updates * args.warmup_ratio
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    best_val_loss = float("inf")
    early_stop_best_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history: list[dict[str, object]] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        running_micro_batches = 0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(
            train_loader,
            start=1,
        ):
            if not batch:
                continue

            remaining = num_batches - (
                batch_idx - 1
            )
            window_size = min(
                args.grad_accum,
                remaining,
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"]
                scaled_loss = (
                    loss / window_size
                )

            scaled_loss.backward()

            running_loss += float(
                loss.item()
            )
            running_micro_batches += 1

            should_step = (
                batch_idx % args.grad_accum == 0
                or batch_idx == num_batches
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

                if (
                    global_step
                    % args.log_every_updates
                    == 0
                    or global_step == 1
                ):
                    denom = max(
                        running_micro_batches,
                        1,
                    )
                    avg_loss = (
                        running_loss
                        / denom
                    )

                    print(
                        f"epoch={epoch} "
                        f"update={global_step}/{total_updates} "
                        f"train_loss={avg_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}"
                    )

                    history.append(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "phase": "train",
                            "loss": avg_loss,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )

                    running_loss = 0.0
                    running_micro_batches = 0

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device_type="cuda",
        )

        val_loss = float(
            val_metrics["loss"]
        )

        print(
            f"end epoch {epoch}: "
            f"val_loss={val_loss:.4f} "
            f"val_ppl={val_metrics['perplexity']:.3f}"
        )

        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "phase": "val",
                **val_metrics,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                output_dir=output_dir, tag="best", epoch=epoch,
                global_step=global_step, best_val_loss=best_val_loss, args=args,
            )
        if val_loss < early_stop_best_loss - args.early_stopping_min_delta:
            early_stop_best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            output_dir=output_dir, tag="last", epoch=epoch,
            global_step=global_step, best_val_loss=best_val_loss, args=args,
        )

        with (output_dir / "history.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                history,
                f,
                indent=2,
                ensure_ascii=False,
            )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement
            >= args.early_stopping_patience
            and epoch
            >= args.early_stopping_start_epoch
        ):
            print(
                "Early stopping triggered: "
                f"{epochs_without_improvement} epochs without improvement."
            )
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        default=str(PREFIX_OUTPUT_DIR),
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
        "--num_virtual_tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--prefix_projection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--prefix_projection_hidden_size",
        type=int,
        default=512,
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
    )
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
        "--early_stopping_patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
    )
    parser.add_argument(
        "--early_stopping_start_epoch",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--log_every_updates",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
