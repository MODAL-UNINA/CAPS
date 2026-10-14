"""Model loading and generation for the unified evaluation CLI."""
from contextlib import nullcontext
import json
from pathlib import Path

from utils.config import DEFAULT_MODEL, DEFAULT_CAPTION_PROMPT, ASSISTANT_MARKER
from utils.utils import format_caption_prompt


def read_run_config(checkpoint):
    if checkpoint is None:
        return {}
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for parent in (directory, directory.parent, directory.parent.parent):
        for name in ('run_config.json', 'args.json'):
            path = parent / name
            if path.is_file():
                return json.loads(path.read_text(encoding='utf-8'))
    return {}


def resolve_checkpoint(checkpoint, method):
    path = Path(checkpoint).expanduser().resolve()
    if method in ('caps_cultural_loss', 'image_conditioned_prompt_tuning'):
        path = path / 'image_conditioned_prompt.pt' if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, {}
    for directory in (path, path / 'adapter'):
        config_file = directory / 'adapter_config.json'
        if config_file.is_file():
            config = json.loads(config_file.read_text(encoding='utf-8'))
            expected = 'PROMPT_TUNING' if method == 'static_prompt_tuning' else 'PREFIX_TUNING'
            if config.get('peft_type') != expected:
                raise ValueError(f'Expected {expected}, found {config.get("peft_type")}')
            return directory, config
    raise FileNotFoundError(f'No adapter_config.json under {path}')


def decode_new_tokens(ids, tokenizer, input_ids=None):
    import torch
    tokens = ids[0]
    if input_ids is not None:
        source = input_ids[0].to(tokens.device)
        if tokens.numel() >= source.numel() and torch.equal(tokens[:source.numel()], source):
            tokens = tokens[source.numel():]
    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return text[len(ASSISTANT_MARKER):].strip() if text.startswith(ASSISTANT_MARKER) else text


class CaptionGenerator:
    def __init__(self, args):
        import torch
        self.torch, self.method = torch, args.method
        self.device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        config = read_run_config(args.checkpoint)
        model_name = args.model_name or config.get('model_name', DEFAULT_MODEL)
        quantized = args.load_in_4bit if args.load_in_4bit is not None else config.get('load_in_4bit', True)
        if quantized and self.device.type != 'cuda':
            raise ValueError('4-bit generation requires CUDA; use --no-load-in-4bit on CPU')
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device.type == 'cuda' else torch.float32
        self.prompt = args.caption_prompt or config.get('caption_prompt', DEFAULT_CAPTION_PROMPT)
        common = dict(model_name=model_name, load_in_4bit=quantized, compute_dtype=dtype, device_map=str(self.device))
        if self.method == 'llava_baseline':
            from utils.utils import load_llava
            self.model, self.processor = load_llava(model_name, quantized, dtype, str(self.device))
            self.backbone = self.model
        else:
            checkpoint, adapter = resolve_checkpoint(args.checkpoint, self.method)
            if self.method == 'static_prompt_tuning':
                from prompt_tuning.model import StaticPromptLlava
                self.model = StaticPromptLlava(**common, prompt_length=int(adapter['num_virtual_tokens']),
                    prompt_position=args.prompt_position or config.get('prompt_position', 'after_bos'))
                self.model.load_adapter(checkpoint)
            elif self.method == 'prefix_tuning':
                from prefix_tuning.model import PrefixTuningLlava
                self.model = PrefixTuningLlava(**common, num_virtual_tokens=int(adapter['num_virtual_tokens']),
                    prefix_projection=bool(adapter.get('prefix_projection', False)),
                    prefix_projection_hidden_size=adapter.get('encoder_hidden_size') or 512)
                self.model.load_prefix_adapter(checkpoint)
            else:
                from CAPS.model import ImageConditionedPromptLlava
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                state = saved.get('state_dict', saved)
                self.model = ImageConditionedPromptLlava(**common,
                    prompt_length=int(state['base_prompt'].shape[0]),
                    bottleneck_size=int(state['generator.0.weight'].shape[0]),
                    prompt_dropout=float(config.get('prompt_dropout', .1)),
                    residual_scale=args.residual_scale if args.residual_scale is not None else float(config.get('residual_scale', 1.0)))
                self.model.load_prompt_module(checkpoint)
                self.model.prompt_module.to(self.model.backbone_device)
            self.backbone, self.processor = self.model.backbone, self.model.processor
        self.model.eval()
        self.metadata = dict(model_name=model_name, caption_prompt=self.prompt, load_in_4bit=quantized,
                             checkpoint=str(args.checkpoint) if args.checkpoint else None, device=str(self.device))

    def generate(self, image, max_new_tokens=512, include_baseline=False):
        inputs = self.processor(text=format_caption_prompt(self.prompt), images=image, return_tensors='pt')
        device = self.backbone.get_input_embeddings().weight.device
        vision = next(self.backbone.vision_tower.parameters())
        model_inputs = dict(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device),
                            pixel_values=inputs['pixel_values'].to(device=vision.device, dtype=vision.dtype))
        tokenizer = self.processor.tokenizer
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id)
        with self.torch.inference_mode():
            ids = self.model.generate(**model_inputs, **kwargs)
            captions = {self.method: decode_new_tokens(ids, tokenizer,
                model_inputs['input_ids'] if self.method == 'llava_baseline' else None)}
            if include_baseline and self.method != 'llava_baseline':
                context = self.model.prefix_model.disable_adapter() if self.method == 'prefix_tuning' else nullcontext()
                with context:
                    ids = self.backbone.generate(**model_inputs, **kwargs)
                captions['llava_baseline'] = decode_new_tokens(ids, tokenizer, model_inputs['input_ids'])
        return captions
