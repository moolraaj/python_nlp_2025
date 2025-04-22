import json
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

STOPWORDS = set(stopwords.words('english'))

LEMMATIZER = WordNetLemmatizer()

 
DEFAULT_SCENE = {
    'id': 'default',
    'name': 'default',
    'gif_url': 'https://example.com/default.gif',
    'background_url': 'https://example.com/default.jpg',
    'animation_type': 'none',
    'message': 'This is the default scene.'
}
 
with open("data/animation_types.json", encoding="utf-8") as f:
    ANIM_TYPES = json.load(f)
with open("data/backgrounds.json", encoding="utf-8") as f:
    BACKGROUNDS = json.load(f)
with open("data/gifs.json", encoding="utf-8") as f:
    GIFS = json.load(f)

 
def extract_keywords(text: str):
    tokens = word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    keywords = []
    for word, tag in tagged:
        if tag in ("NN", "VBG") and word.isalpha() and word not in STOPWORDS:
            lemma = LEMMATIZER.lemmatize(word)
            if lemma not in keywords:
                keywords.append(lemma)
    return keywords

 
def find_assets(query: str):
    kws = extract_keywords(query)

    animations = [a for a in ANIM_TYPES if a.get("name") in kws]
    backgrounds = [b for b in BACKGROUNDS if b.get("name") in kws]
    gifs = [g for g in GIFS if any(tag in kws for tag in g.get('tags', []))]

    
    if not (animations or backgrounds or gifs):
        animations = [a for a in ANIM_TYPES if a.get("name") == "default"]
        backgrounds = [b for b in BACKGROUNDS if b.get("name") == "default"]
        gifs = [g for g in GIFS if "default" in g.get("tags", [])]

    return {
        'keywords': kws,
        'animations': animations,
        'backgrounds': backgrounds,
        'gifs': gifs
    }



def suggest_random():
    candidates = [g for g in GIFS if g.get('id') != 'default']
    chosen = random.choice(candidates) if candidates else DEFAULT_SCENE
    tags = chosen.get('tags', [])
    char = tags[0] if tags else 'character'
    anim = next((a['name'] for a in ANIM_TYPES if a['id'] == chosen.get('animation_type_id')), 'walking')
    bg = next((b['name'] for b in BACKGROUNDS if b['id'] == chosen.get('background_id')), 'temple')
    suggestion_text = f"Try something like this -->  a {char} {anim} to the {bg}"
    suggestion_assets = find_assets(suggestion_text)
    return suggestion_text, suggestion_assets

 
