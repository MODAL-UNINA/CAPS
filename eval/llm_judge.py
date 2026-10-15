import json
from pathlib import Path
import time

from utils.config import JUDGE_SYSTEM_PROMPT
from utils.utils import build_judge_user_prompt

def normalize_references(value):
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith('[') else [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError('Three human references are required')
    refs = [text.strip() for text in value if isinstance(text, str) and text.strip()]
    return refs


def parse_judgment(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('Judge response must be an object')
    for key in ('caption_accuracy', 'cultural_relevance'):
        if type(result.get(key)) is not int or not 1 <= result[key] <= 5:
            raise ValueError(f'Invalid {key}')
    for key in ('accuracy_reason', 'cultural_reason'):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f'Missing {key}')
    return {key: result[key] for key in ('caption_accuracy', 'cultural_relevance', 'accuracy_reason', 'cultural_reason')}


class CaptionJudge:
    def __init__(self, model, base_url, attempts=2):
        from ollama import Client
        self.client, self.model, self.attempts = Client(host=base_url), model, attempts

    def score(self, image_path, caption, references):
        refs = normalize_references(references)
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not isinstance(caption, str):
            raise TypeError('Caption must be a string')
        prompt = build_judge_user_prompt(refs, caption)
        schema = {'type': 'object', 'properties': {
            'caption_accuracy': {'type': 'integer', 'minimum': 1, 'maximum': 5},
            'cultural_relevance': {'type': 'integer', 'minimum': 1, 'maximum': 5},
            'accuracy_reason': {'type': 'string', 'minLength': 1},
            'cultural_reason': {'type': 'string', 'minLength': 1}},
            'required': ['caption_accuracy', 'cultural_relevance', 'accuracy_reason', 'cultural_reason']}
        error = None
        for attempt in range(self.attempts):
            try:
                response = self.client.chat(model=self.model,
                    messages=[{'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
                              {'role': 'user', 'content': prompt, 'images': [str(image_path)]}],
                    format=schema, options={'temperature': 0.0})
                message = response['message'] if isinstance(response, dict) else response.message
                return parse_judgment(message['content'] if isinstance(message, dict) else message.content)
            except Exception as exc:
                error = exc
                if attempt+1 < self.attempts:
                    time.sleep(1)
        raise RuntimeError(f'Judge failed after {self.attempts} attempts') from error
