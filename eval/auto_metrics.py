import importlib
import io
from pathlib import Path
import sys

import torch
from PIL import Image
from nltk.corpus import wordnet as wn

clip = None
longclip = None


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def truncate_text(text, max_length):
    if len(text) > max_length:
        truncated = text[:max_length].rsplit(' ', 1)[0]
        if not truncated.endswith('.'):
            truncated += '.'
        return truncated
    return text if text.endswith('.') else text + '.'


# ============================================================
# LONGCLIP SCORE
# ============================================================

def get_longclip_score(image_path, text, model, preprocess, first_sentence_only=False, max_length=248):
    global longclip
    if longclip is None:
        from LONGCLIP.model import longclip

    image = Image.open(image_path)

    if not isinstance(text, str):
        print(f"Warning: Expected text but got {type(text)} at {image_path}. Defaulting to empty string.")
        text = ""

    if first_sentence_only:
        text = text.split('.')[0].strip()

    text = truncate_text(text, max_length)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = longclip.tokenize([text]).to(device)


    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    longclip_score = torch.matmul(image_features, text_features.T).item()
    return longclip_score


# ============================================================
# CULTURAL VOCABULARY AND METRICS
# ============================================================

def load_cultural_words(filename):
    try:
        with open(filename, 'r') as file:
            cultural_words = {line.strip().lower() for line in file}
        return cultural_words
    except IOError:
        print("Error: File does not appear to exist.")
        return set()


def calculate_cnr(caption, cultural_words):
    if not isinstance(caption, str):
        print(f"Warning: Expected text but got {type(caption)}. Defaulting to empty string.")
        caption = ""

    words = set(caption.lower().split())
    cultural_count = sum(1 for word in words if word in cultural_words)
    return cultural_count / len(words) if words else 0


def calculate_cultural_number(caption, cultural_words):
    if not isinstance(caption, str):
        print(f"Warning: Expected text but got {type(caption)}. Defaulting to empty string.")
        caption = ""

    words = set(caption.lower().split())
    cultural_count = sum(1 for word in words if word in cultural_words)
    return cultural_count if words else 0


# ============================================================
# WORDNET AND COMPLETENESS
# ============================================================

def get_wordnet_related_words(word):
    related_words = set()
    for syn in wn.synsets(word):
        # Add the synonyms
        related_words.update(syn.lemma_names())
        # Add the hyponyms
        related_words.update(lem.name() for hypon in syn.hyponyms() for lem in hypon.lemmas())
        # Add the hypernyms
        related_words.update(lem.name() for hyper in syn.hypernyms() for lem in hyper.lemmas())
    # Normalize and remove underscores
    return set(w.replace('_', ' ').lower() for w in related_words)


def calculate_completeness(tag_string, caption_string):

    if not isinstance(caption_string, str):
        print(f"Warning: Expected text but got {type(caption_string)}. Defaulting to empty string.")
        caption_string = ""
    
    tags = set(tag_string.split(", "))
    expanded_tags = set()
    # Expand each tag with WordNet related words
    for tag in tags:
        expanded_tags.update(get_wordnet_related_words(tag))
    caption_words = set(caption_string.lower().split())
    # Calculate the intersection of expanded tags and caption words
    intersection = expanded_tags.intersection(caption_words)
    # Return the ratio of matched tags to total tags
    if len(tags) == 0:
        return 0
    return len(intersection) / len(tags)


# ============================================================
# EVALUATION ADAPTER
# ============================================================

class Metrics:

    def __init__(self, cultural_words, longclip_repo, longclip_checkpoint):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vocabulary = load_cultural_words(cultural_words)
        checkpoint = Path(longclip_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        repo = Path(longclip_repo).resolve()
        if (repo / 'LONGCLIP' / 'model').is_dir():
            module = 'LONGCLIP.model.longclip'
        elif (repo / 'model').is_dir():
            module = 'model.longclip'
        else:
            raise ValueError(f'No LongCLIP model package under {repo}')
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        self.longclip = importlib.import_module(module)
        self.model, self.preprocess = self.longclip.load(str(checkpoint), device=device)
        self.model.eval()

    def score(self, image, caption, tags):
        global device, longclip
        device = self.device
        longclip = self.longclip
        if isinstance(image, Image.Image):
            with io.BytesIO() as stream:
                image.save(stream, format='TIFF')
                stream.seek(0)
                alignment = get_longclip_score(stream, caption, self.model, self.preprocess)
        else:
            alignment = get_longclip_score(image, caption, self.model, self.preprocess)
        return {
            'alignment': alignment,
            'completeness': calculate_completeness(tags, caption),
            'cultural_information': calculate_cultural_number(caption, self.vocabulary),
            'cnr': calculate_cnr(caption, self.vocabulary),
            'avg_length': len(caption.split()) if isinstance(caption, str) else 0,
        }
