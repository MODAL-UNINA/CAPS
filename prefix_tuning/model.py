from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import (
    PeftModel,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
)

from utils.config import DEFAULT_MODEL
from utils.utils import FrozenVisionMixin, load_llava


class PrefixTuningLlava(FrozenVisionMixin, nn.Module):
    """Frozen LLaVA-1.5 with a standard PEFT Prefix-Tuning language model."""


    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        num_virtual_tokens: int = 16,
        prefix_projection: bool = False,
        prefix_projection_hidden_size: int = 512,
        load_in_4bit: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] = "auto",
    ) -> None:
        super().__init__()

        self.backbone, self.processor = load_llava(
            model_name, load_in_4bit, compute_dtype, device_map,
        )

        # Standard PEFT Prefix Tuning is applied ONLY to the language model.
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prefix_projection=prefix_projection,
            encoder_hidden_size=(
                prefix_projection_hidden_size
                if prefix_projection
                else None
            ),
            inference_mode=False,
        )

        self.prefix_model: PeftModel = get_peft_model(
            self.backbone.language_model,
            prefix_config,
        )

        for name, parameter in self.prefix_model.named_parameters():
            if parameter.requires_grad:
                nn.init.zeros_(parameter)

        self.backbone.language_model = self.prefix_model
        self.num_virtual_tokens = num_virtual_tokens


    def train(self, mode: bool = True):
        """Train the PEFT prefix; keep the frozen LLaVA backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        self.prefix_model.train(mode)
        return self

    @property
    def backbone_device(self) -> torch.device:
        return next(
            self.backbone.language_model.parameters()
        ).device

    def save_prefix_adapter(
        self,
        output_dir: str | Path,
    ) -> None:
        """
        Save only the PEFT prefix adapter/configuration.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.prefix_model.save_pretrained(
            output_dir,
            safe_serialization=True,
        )

    def load_prefix_adapter(
        self,
        adapter_dir: str | Path,
        is_trainable: bool = False,
    ) -> None:
        """
        Load a previously saved PEFT prefix adapter.
        """
        adapter_dir = Path(adapter_dir)

        base_lm = self.backbone.language_model
        if isinstance(base_lm, PeftModel):
            self.prefix_model.load_adapter(
                adapter_dir,
                adapter_name="default",
                is_trainable=is_trainable,
            )
            self.prefix_model.set_adapter("default")
            return

        raise RuntimeError(
            "Unexpected language_model type: expected PeftModel."
        )


    def build_multimodal_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Build the standard LLaVA multimodal input sequence.

        """
        language_device = self.backbone_device

        input_ids = input_ids.to(language_device)
        attention_mask = attention_mask.to(language_device)

        image_features = self._encode_images(
            pixel_values
        ).to(language_device)

        token_embeddings = self.backbone.get_input_embeddings()(
            input_ids
        )

        token_embeddings = self._replace_image_placeholders(
            input_ids=input_ids,
            token_embeddings=token_embeddings,
            image_features=image_features,
        )

        multimodal_labels = None
        if labels is not None:
            multimodal_labels = labels.to(
                language_device
            )

        return (
            token_embeddings,
            attention_mask,
            multimodal_labels,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Teacher-forced captioning with standard PEFT Prefix Tuning.
        """
        (
            inputs_embeds,
            multimodal_mask,
            multimodal_labels,
        ) = self.build_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        outputs = self.prefix_model(
            inputs_embeds=inputs_embeds,
            attention_mask=multimodal_mask,
            labels=multimodal_labels,
            use_cache=False,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate a caption with the frozen multimodal backbone + learned
        prefix adapter.
        """
        (
            inputs_embeds,
            multimodal_mask,
            _,
        ) = self.build_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        generation_kwargs.setdefault(
            "max_new_tokens",
            128,
        )
        generation_kwargs.setdefault(
            "do_sample",
            False,
        )
        generation_kwargs.setdefault(
            "pad_token_id",
            self.processor.tokenizer.eos_token_id,
        )
        generation_kwargs.setdefault(
            "eos_token_id",
            self.processor.tokenizer.eos_token_id,
        )


        prefix_cache = self.prefix_model.get_prompt(batch_size=inputs_embeds.size(0))
        cached_slots = inputs_embeds.new_zeros(
            inputs_embeds.size(0), self.num_virtual_tokens, inputs_embeds.size(-1)
        )
        prefix_mask = multimodal_mask.new_ones(
            inputs_embeds.size(0), self.num_virtual_tokens
        )
        return self.prefix_model.get_base_model().generate(
            inputs_embeds=torch.cat((cached_slots, inputs_embeds), dim=1),
            attention_mask=torch.cat((prefix_mask, multimodal_mask), dim=1),
            past_key_values=prefix_cache,
            **generation_kwargs,
        )
