from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    PeftModel,
)

from utils.config import DEFAULT_MODEL
from utils.utils import FrozenVisionMixin, load_llava


class StaticPromptLlava(FrozenVisionMixin, nn.Module):

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prompt_length: int = 16,
        load_in_4bit: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "cuda:0",
        init_prompt_text: str | None = None,
        prompt_position: str = "after_bos",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.prompt_length = prompt_length
        if prompt_position not in {"after_bos", "prepend"}:
            raise ValueError("prompt_position must be 'after_bos' or 'prepend'")
        self.prompt_position = prompt_position

        self.backbone, self.processor = load_llava(
            model_name, load_in_4bit, compute_dtype, device_map,
        )

        if load_in_4bit:
            self.quantization_method = getattr(self.backbone, "quantization_method", None)
            self.is_loaded_in_4bit = True
            self.hf_device_map = getattr(self.backbone, "hf_device_map", None)


        config_kwargs = dict(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=prompt_length,
            prompt_tuning_init=(
                PromptTuningInit.TEXT if init_prompt_text else PromptTuningInit.RANDOM
            ),
        )
        if init_prompt_text:
            config_kwargs.update(
                prompt_tuning_init_text=init_prompt_text,
                tokenizer_name_or_path=model_name,
            )
        peft_config = PromptTuningConfig(**config_kwargs)


        self.peft_lm = get_peft_model(self.backbone.language_model, peft_config)
        self.peft_lm.prompt_encoder[
            self.peft_lm.active_adapter
        ].to(device=self.language_device, dtype=torch.bfloat16)

        self.backbone.config.use_cache = False
        self.backbone.language_model.config.use_cache = False

    @property
    def language_device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    def save_adapter(self, directory: str | Path) -> None:
        """Save the PEFT adapter (adapter_config.json and adapter_model.safetensors)."""
        self.peft_lm.save_pretrained(str(directory), safe_serialization=True)

    def load_adapter(self, directory: str | Path) -> None:
        """Load a saved PEFT adapter around the frozen language model.

        Intended for inference, not for resuming optimization.
        """
        self.peft_lm = PeftModel.from_pretrained(
            self.backbone.language_model, str(directory), is_trainable=False
        )

    def save_prompt_state(self, path: str | Path) -> None:
        """Save the minimal prompt state used by Trainer for resume and best model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        weight = self.peft_lm.prompt_encoder[
            self.peft_lm.active_adapter
        ].embedding.weight
        torch.save(weight.detach().cpu(), path)

    def load_prompt_state(self, path: str | Path) -> None:
        """Copy in place to preserve the optimizer's parameter references."""
        weight = torch.load(path, map_location="cpu", weights_only=True)
        dest = self.peft_lm.prompt_encoder[
            self.peft_lm.active_adapter
        ].embedding.weight
        if tuple(weight.shape) != tuple(dest.shape):
            raise ValueError(f"Soft prompt: shape {tuple(weight.shape)} != {tuple(dest.shape)}")
        with torch.no_grad():
            dest.copy_(weight.to(device=dest.device, dtype=dest.dtype))

    def _multimodal_embeds(self, pixel_values: torch.Tensor,
                           input_ids: torch.Tensor) -> torch.Tensor:
        """Build frozen text and image embeddings before adding soft prompts."""
        input_ids = input_ids.to(self.language_device)
        with torch.no_grad():
            image_features = self._encode_images(pixel_values)
            text_embeds = self.backbone.get_input_embeddings()(input_ids)
            return self._replace_image_placeholders(input_ids, text_embeds, image_features)

    def build_extended_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        text_embeds = self._multimodal_embeds(pixel_values, input_ids)
        batch_size = text_embeds.size(0)

        prompt = self.peft_lm.get_prompt(batch_size=batch_size).to(
            device=text_embeds.device, dtype=text_embeds.dtype
        )

        if self.prompt_position == "after_bos":
            bos_id = self.processor.tokenizer.bos_token_id
            if bos_id is None or not torch.all(
                input_ids[:, 0].to(text_embeds.device) == bos_id
            ):
                raise ValueError(
                    "prompt_position='after_bos' requires BOS as the first token "
                    "of every example.
                )
            split = 1
        else:
            split = 0

        inputs_embeds = torch.cat(
            (text_embeds[:, :split], prompt, text_embeds[:, split:]), dim=1
        )
        attention_mask = attention_mask.to(text_embeds.device)
        prefix_attention = torch.ones(
            (batch_size, self.prompt_length),
            dtype=attention_mask.dtype,
            device=text_embeds.device,
        )
        extended_mask = torch.cat(
            (attention_mask[:, :split], prefix_attention, attention_mask[:, split:]),
            dim=1,
        )
        extended_labels = None
        if labels is not None:
            labels = labels.to(text_embeds.device)
            prefix_labels = torch.full(
                (batch_size, self.prompt_length),
                -100,
                dtype=labels.dtype,
                device=text_embeds.device,
            )
            extended_labels = torch.cat(
                (labels[:, :split], prefix_labels, labels[:, split:]), dim=1
            )
        return inputs_embeds, extended_mask, extended_labels

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        embeds, mask, extended_labels = self.build_extended_inputs(
            pixel_values, input_ids, attention_mask, labels
        )
        outputs = self.backbone.language_model(
            inputs_embeds=embeds,
            attention_mask=mask,
            labels=extended_labels,
            use_cache=False,
            return_dict=True,
        )
        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        embeds, mask, _ = self.build_extended_inputs(
            pixel_values, input_ids, attention_mask
        )
        kwargs.setdefault("max_new_tokens", 128)
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("pad_token_id", self.processor.tokenizer.pad_token_id)
        kwargs.setdefault("eos_token_id", self.processor.tokenizer.eos_token_id)
        return self.backbone.language_model.generate(
            inputs_embeds=embeds, attention_mask=mask, **kwargs
        )
