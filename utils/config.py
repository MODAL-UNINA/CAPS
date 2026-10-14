"""Shared configuration for CAPS data preparation, training and evaluation."""
from pathlib import Path

# ============================================================
# MODEL IDENTIFIERS
# ============================================================
DEFAULT_MODEL = 'llava-hf/llava-1.5-7b-hf'
CAPTION_MODEL = 'hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_XL'
JUDGE_MODEL = 'hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0'


# ============================================================
# DIRECTORY AND FILE PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('data')
DEFAULT_IMAGES_ROOT = DATA_ROOT
DEFAULT_SOURCE_JSON = DATA_ROOT / 'CulturalGROUND/CulturalGround-OE-Filtered-14M.json'
POSITIVE_OUTPUT_DIR = DATA_ROOT / 'positive_captions'
NEGATIVE_OUTPUT_DIR = DATA_ROOT / 'negative_captions'
SPLIT_OUTPUT_DIR = DATA_ROOT / 'splits'
PROMPT_OUTPUT_DIR = Path('checkpoints/prompt_tuning')
PREFIX_OUTPUT_DIR = Path('checkpoints/prefix_tuning')
CAPS_OUTPUT_DIR = Path('checkpoints/caps')
CULTURAL_WORDS_DIR = PROJECT_ROOT / 'eval' / 'cultural_words'
CULTURAL_WORDS_FILE = PROJECT_ROOT / 'eval' / 'cultural_words' / 'all_cultural_words.txt'
HUMAN_CAPTIONS_DIR = PROJECT_ROOT / 'eval' / 'human_metrics' / 'HumanCaptions'


# ============================================================
# IMAGE PATH MAPPING
# ============================================================
LEGACY_IMAGE_PREFIX = 'Wikidata_images_v3/'
IMAGE_PREFIX = 'CulturalGROUND/CultureGroundImages/'


# ============================================================
# LOCAL MODEL SERVER
# ============================================================
OLLAMA_BASE_URL = 'http://localhost:11434'


# ============================================================
# DATA PIPELINE CONFIGURATION
# ============================================================
DEFAULT_SEED = 42
VALIDATION_RATIO = 0.1
TRAIN_SPLIT = 'train_sampled'
VAL_SPLIT = 'val_sampled'
MAX_IMAGE_SIDE = 2048


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
LEARNING_RATE = 1e-4
EARLY_STOPPING_MIN_DELTA = 1e-4


# ============================================================
# IMAGE-CONDITIONED PROMPT CONFIGURATION
# ============================================================
PROMPT_BOTTLENECK_SIZE = 256


# ============================================================
# EVALUATION METRICS
# ============================================================
LONGCLIP_MAX_CHARACTERS = 248
METRIC_NAMES = ('alignment', 'completeness', 'cultural_information', 'cnr', 'avg_length')
JUDGE_METRIC_NAMES = ('caption_accuracy', 'cultural_relevance')


# ============================================================
# EVALUATION METHODS AND CAPTION FIELDS
# ============================================================
METHOD_FIELDS = {
    'llava_baseline': 'caption_llava_baseline',
    'prefix_tuning': 'caption_prefix_tuning',
    'static_prompt_tuning': 'caption_static_prompt_tuning',
    'image_conditioned_prompt_tuning': 'caption_image_conditioned_prompt_tuning',
    'caps_cultural_loss': 'caption_caps_cultural_loss',
}

METHOD_LABELS = {
    'llava_baseline': 'Frozen LLaVA baseline',
    'prefix_tuning': 'Prefix tuning',
    'static_prompt_tuning': 'Static prompt tuning',
    'image_conditioned_prompt_tuning': 'Image-conditioned prompt tuning (caption loss only)',
    'caps_cultural_loss': 'CAPS (caption loss + cultural span loss)',
}


# ============================================================
# CAPTIONING INSTRUCTION
# ============================================================
DEFAULT_CAPTION_PROMPT = "Describe the image in detail."
ASSISTANT_MARKER = "ASSISTANT:"


# ============================================================
# POSITIVE CAPTION STYLES
# ============================================================
CAPTION_STYLES = {
    "neutral": """
Write a natural descriptive caption without using a fixed opening.
""",

    "subject_first": """
Begin with the main visible subject, object, or activity.
""",

    "scene_first": """
Begin with the visible setting or overall scene, then describe the main subject.
""",

    "image_shows": """
Begin with "The image shows" and continue with a natural description.
""",
    "image_depicts": """
Begin with "The image depicts" and continue with a natural description.
""",
}

STYLE_WEIGHTS = [
    ("neutral", 20),
    ("subject_first", 20),
    ("scene_first", 20),
    ("image_shows", 20),
    ("image_depicts", 20),
]


# ============================================================
# POSITIVE CAPTION SYSTEM PROMPT
# ============================================================
CAPTION_SYSTEM_PROMPT = """
You are generating training captions for a culturally aware image-captioning model.

You receive an image, a cultural question-answer pair, country metadata, and a
requested caption style.

Write one natural English caption that is both visually grounded and culturally
informative.

CONTENT PRIORITY
1. Describe the visible image accurately.
2. Use the QA only to understand and name visually supported cultural content.
3. Prefer specific visible cultural terms, such as the name of a garment, food,
   architectural type, artwork, symbol, ritual object, or landmark, when
   justified by the image.
4. Use an exact person, city, event, institution, or location name only when the
   image contains distinctive evidence that could support that identity.
5. Otherwise, replace the exact name with an accurate descriptive category,
   such as "a political figure", "an urban waterfront", "a religious building",
   or "a formal public gathering".
6. If no culturally specific detail is visually supportable, write a normal
   accurate visual caption. Do not force cultural content.

GROUNDING
7. Do not copy dates, biographies, terms of office, population, nicknames,
   economic facts, ownership, diplomatic relationships, causes, consequences,
   or unrelated historical information.
8. Do not infer ethnicity, religion, emotion, intention, or political
   affiliation from appearance alone.
9. Do not mention the QA, metadata, dataset, uncertainty assessment, or
   requested formulation.

FORM
10. Follow the requested linguistic formulation naturally.
11. Let the caption length depend on the amount of visually grounded information
    available. Use one to three natural sentences and usually about 30 to 90
    words. Shorter or longer is acceptable when warranted by the image. Do not
    add facts merely to reach a target length.
12. Output only the final caption.
""".strip()


# ============================================================
# LLM JUDGE SYSTEM PROMPT
# ============================================================
JUDGE_SYSTEM_PROMPT = """
You are evaluating a model-generated image caption against three
human-written reference captions.

You are given:
1. the image;
2. a candidate caption generated by a model;
3. three human-written reference captions.

Evaluate the candidate caption on the following two criteria.

1. Caption Accuracy
Evaluate how accurately the candidate caption reflects the content
described by the human reference captions.
Score:
1 = largely incorrect or inconsistent with the references
2 = several important inaccuracies
3 = mostly correct, with some missing or inaccurate details
4 = accurate, with only minor issues
5 = highly accurate and consistent with the human references

2. Cultural Relevance
Evaluate how well the candidate caption captures the culturally
relevant information expressed in the human reference captions.
Score:
1 = major cultural misidentification or no relevant cultural content
2 = very limited or substantially inaccurate cultural content
3 = some correct cultural content, but generic, incomplete, or partly inaccurate
4 = accurate and specific cultural content, with minor omissions or issues
5 = accurately captures the important cultural information expressed
in the human references

Cultural details are scene-relevant elements such as traditional clothing, food, architecture, artifacts, symbols, or practices. 

Return JSON only:
{
  "caption_accuracy": 1-5,
  "cultural_relevance": 1-5,
  "accuracy_reason": "Brief explanation referring to specific claims.",
  "cultural_reason": "Brief explanation of the cultural content captured, omitted, or misidentified."
}""".strip()

