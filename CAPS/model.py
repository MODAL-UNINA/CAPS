from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from utils.config import DEFAULT_MODEL, PROMPT_BOTTLENECK_SIZE
from utils.utils import FrozenVisionMixin, load_llava


class ImageConditionedSoftPrompt(nn.Module):
    """Generate one continuous prompt per image from pooled visual features."""

    def __init__(
        self,
        prompt_length: int,
        hidden_size: int,
        bottleneck_size: int = PROMPT_BOTTLENECK_SIZE,
        dropout: float = 0.1,
        init_std: float = 0.02,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.residual_scale = residual_scale

        self.base_prompt = nn.Parameter(
            torch.empty(prompt_length, hidden_size)
        )
        nn.init.normal_(self.base_prompt, mean=0.0, std=init_std)

        self.visual_norm = nn.LayerNorm(hidden_size)
        self.generator = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                bottleneck_size,
                prompt_length * hidden_size,
            ),
        )

        output_layer = self.generator[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: projected visual patches [B, N, D].

        Returns:
            Image-conditioned prompt embeddings [B, L, D].
        """
        pooled = image_features.mean(dim=1)
        pooled = self.visual_norm(pooled)
        delta = self.generator(pooled)
        delta = delta.view(
            image_features.size(0),
            self.prompt_length,
            self.hidden_size,
        )
        return self.base_prompt.unsqueeze(0) + self.residual_scale * delta


class ImageConditionedPromptLlava(FrozenVisionMixin, nn.Module):
    """Frozen LLaVA backbone with a trainable image-conditioned input prompt."""


    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prompt_length: int = 16,
        bottleneck_size: int = PROMPT_BOTTLENECK_SIZE,
        prompt_dropout: float = 0.1,
        residual_scale: float = 1.0,
        load_in_4bit: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] = "auto",
    ) -> None:
        super().__init__()

        self.backbone, self.processor = load_llava(
            model_name, load_in_4bit, compute_dtype, device_map,
        )


        hidden_size = self.backbone.language_model.config.hidden_size
        self.prompt_module = ImageConditionedSoftPrompt(
            prompt_length=prompt_length,
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            dropout=prompt_dropout,
            residual_scale=residual_scale,
        )
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size

    def train(self, mode: bool = True):
        """Train only the prompt module; keep the frozen backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        self.prompt_module.train(mode)
        return self

    @property
    def backbone_device(self) -> torch.device:
        return next(self.backbone.language_model.parameters()).device

    @property
    def prompt_device(self) -> torch.device:
        return self.prompt_module.base_prompt.device

    def initialize_base_prompt_from_text(self, text: str) -> None:
        """Initialize the shared base prompt from natural-language embeddings."""
        tokenizer = self.processor.tokenizer
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("Initialization text produced no tokens.")

        token_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.backbone_device,
        )

        with torch.no_grad():
            text_embeddings = self.backbone.get_input_embeddings()(token_tensor)
            if text_embeddings.size(0) < self.prompt_length:
                repeats = (
                    self.prompt_length + text_embeddings.size(0) - 1
                ) // text_embeddings.size(0)
                text_embeddings = text_embeddings.repeat(repeats, 1)

            text_embeddings = text_embeddings[: self.prompt_length]
            self.prompt_module.base_prompt.copy_(
                text_embeddings.to(
                    device=self.prompt_device,
                    dtype=self.prompt_module.base_prompt.dtype,
                )
            )

    def initialize_base_prompt_from_static_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        """Warm-start the base prompt from your previous static-prompt checkpoint."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Existing static checkpoints save `embedding`; tolerate full names too.
        candidate_keys = (
            "embedding",
            "soft_prompt.embedding",
            "base_prompt",
            "prompt_module.base_prompt",
        )
        source = None
        for key in candidate_keys:
            if key in state_dict:
                source = state_dict[key]
                break

        if source is None:
            raise KeyError(
                "Could not find the static prompt tensor in the checkpoint. "
                f"Available keys: {list(state_dict.keys())}"
            )
        if tuple(source.shape) != tuple(self.prompt_module.base_prompt.shape):
            raise ValueError(
                f"Prompt shape mismatch: checkpoint has {tuple(source.shape)}, "
                f"model expects {tuple(self.prompt_module.base_prompt.shape)}."
            )

        with torch.no_grad():
            self.prompt_module.base_prompt.copy_(
                source.to(
                    device=self.prompt_device,
                    dtype=self.prompt_module.base_prompt.dtype,
                )
            )

    def save_prompt_module(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_length": self.prompt_length,
                "hidden_size": self.hidden_size,
                "state_dict": self.prompt_module.state_dict(),
            },
            output_path,
        )

    def load_prompt_module(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.prompt_module.load_state_dict(state_dict)


    def build_extended_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build [P(image); original multimodal sequence]."""
        language_device = self.backbone_device
        input_ids = input_ids.to(language_device)
        attention_mask = attention_mask.to(language_device)

        image_features = self._encode_images(pixel_values)
        image_features_language = image_features.to(language_device)

        token_embeddings = self.backbone.get_input_embeddings()(input_ids)
        token_embeddings = self._replace_image_placeholders(
            input_ids=input_ids,
            token_embeddings=token_embeddings,
            image_features=image_features_language,
        )

        prompt_input = image_features_language.to(
            device=self.prompt_device,
            dtype=self.prompt_module.base_prompt.dtype,
        )
        prompt_embeddings = self.prompt_module(prompt_input).to(
            device=language_device,
            dtype=token_embeddings.dtype,
        )

        extended_embeddings = torch.cat(
            [prompt_embeddings, token_embeddings],
            dim=1,
        )

        batch_size = input_ids.size(0)
        prompt_mask = torch.ones(
            batch_size,
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
                (batch_size, self.prompt_length),
                -100,
                dtype=labels.dtype,
                device=language_device,
            )
            extended_labels = torch.cat(
                [prompt_labels, labels],
                dim=1,
            )

        return extended_embeddings, extended_attention_mask, extended_labels

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        inputs_embeds, extended_mask, extended_labels = self.build_extended_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        outputs = self.backbone.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=extended_labels,
            use_cache=False,
            return_dict=True,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        inputs_embeds, extended_mask, _ = self.build_extended_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        generation_kwargs.setdefault("max_new_tokens", 128)
        generation_kwargs.setdefault("do_sample", False)
        generation_kwargs.setdefault(
            "pad_token_id",
            self.processor.tokenizer.pad_token_id,
        )
        generation_kwargs.setdefault(
            "eos_token_id",
            self.processor.tokenizer.eos_token_id,
        )

        return self.backbone.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            **generation_kwargs,
        )
