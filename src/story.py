"""Story -> scene breakdown.

Narration is split deterministically in Python (so it is always a faithful,
verbatim excerpt of the input story - never hallucinated). A local LLM
(transformers, GPU fp16 if available) is only asked for the two creative
pieces that are safe to hallucinate-check visually: an English character
description and a short per-scene visual prompt. This hybrid design avoids
relying on a small 1.5B model to reproduce long Hindi text verbatim, which
it does not do reliably.
"""
import gc
import json
import re

from src import config

_model = None
_tokenizer = None

# Small models (1.5B-3B) repeatedly mistranslated the main character's species
# from Hindi (e.g. "बिल्ली" [cat] -> "cow", likely because "milk shop" stories
# are a common cow-themed pattern in training data that overrides the actual
# word). Since getting the species wrong derails every downstream image, the
# most common animal/character nouns are resolved deterministically here and
# only the extra visual traits (color, accessories, features) are left to the
# LLM - it is never asked to translate the species itself when a match is found.
KNOWN_CHARACTER_TERMS = {
    "बिल्ली": "cat", "कुत्ता": "dog", "चूहा": "mouse", "गाय": "cow",
    "बकरी": "goat", "बकरा": "goat", "शेर": "lion", "हाथी": "elephant",
    "बंदर": "monkey", "खरगोश": "rabbit", "भालू": "bear", "मुर्गी": "hen",
    "तोता": "parrot", "कौआ": "crow", "मछली": "fish", "भेड़": "sheep",
    "सांप": "snake", "बाघ": "tiger", "लड़की": "girl", "लड़का": "boy",
}


def detect_character_species(story_text: str):
    """Deterministic lookup; returns (hindi_word, english_species) or None."""
    best = None
    for hindi_word, species in KNOWN_CHARACTER_TERMS.items():
        idx = story_text.find(hindi_word)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, hindi_word, species)
    if best is None:
        return None
    return best[1], best[2]


def load_llm():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = str(config.MODEL_CACHE_DIR / "huggingface")
    _tokenizer = AutoTokenizer.from_pretrained(config.LLM_REPO, cache_dir=cache_dir)
    # Kept on CPU deliberately: at 3B params, fp16-on-GPU would eat most of the
    # 6GB card's VRAM budget that image generation needs right after this step.
    # This stage runs once per video and correctness matters more than speed,
    # so CPU (float32, 64GB RAM available) is the safer trade-off here.
    _model = AutoModelForCausalLM.from_pretrained(
        config.LLM_REPO, cache_dir=cache_dir, dtype=torch.float32,
    )
    _model.eval()
    return _model, _tokenizer


def unload_llm():
    global _model, _tokenizer
    if _model is not None:
        import torch
        del _model
        _model = None
        _tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def split_story_into_scenes(story_text: str, target_scenes: int) -> list:
    """Deterministic sentence split. Never invents or drops text."""
    story_text = story_text.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[।.!?])\s+", story_text) if s.strip()]
    if not sentences:
        return [story_text]
    if len(sentences) <= target_scenes:
        return sentences  # one sentence per scene; natural count may be < target
    # more sentences than target scenes: group evenly, keep sentences intact
    groups = [[] for _ in range(target_scenes)]
    for i, s in enumerate(sentences):
        groups[i * target_scenes // len(sentences)].append(s)
    return [" ".join(g).strip() for g in groups if g]


PROMPT_TEMPLATE = """You are a visual storyboard artist. Here is a short Hindi children's story, already split into {n} scenes in order:

{scene_list}

Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{{
  "title": "short English title for the whole story",
  "character_description": "short English visual description (comma-separated visual traits only: species/type, color, key features) of the SINGLE main character that appears in these scenes - infer the species/type correctly from the story text",
  "scene_visual_prompts": [
    "short English description (5-12 words) of ONLY the action/setting in scene 1, no character traits",
    "short English description (5-12 words) of ONLY the action/setting in scene 2, no character traits"
  ]
}}

Rules:
- "scene_visual_prompts" must have exactly {n} items, in order, one per scene above.
- Do not repeat the character description inside each scene_visual_prompts item.
- Output JSON only, nothing else.
"""

# Used instead of PROMPT_TEMPLATE when detect_character_species() finds a
# confident match: the species is fixed in code, the LLM only adds extra traits.
PROMPT_TEMPLATE_KNOWN_SPECIES = """You are a visual storyboard artist. Here is a short Hindi children's story, already split into {n} scenes in order. The main character is a {species} ("{hindi_word}" in the Hindi text) - this is already known, do not change or restate the species.

{scene_list}

Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{{
  "title": "short English title for the whole story",
  "character_extra_traits": "short comma-separated list of ADDITIONAL visual traits only (color, size, accessories, notable features) for the {species} - do NOT mention the species/animal type itself, just the extra descriptive traits",
  "scene_visual_prompts": [
    "short English description (5-12 words) of ONLY the {species}'s own action/pose/expression and the setting in scene 1",
    "short English description (5-12 words) of ONLY the {species}'s own action/pose/expression and the setting in scene 2"
  ]
}}

Rules:
- "scene_visual_prompts" must have exactly {n} items, in order, one per scene above.
- Each scene_visual_prompts item must describe ONLY the single {species} - never introduce or mention any other character (no shopkeeper, no people, no other animals). If the story sentence involves another character, describe it only through the {species}'s own pose/expression/reaction (e.g. "looking up hopefully" instead of "shopkeeper hands it milk").
- Refer to the character only as "the {species}" in English. Never use Hindi words, transliterations, or invented names (e.g. never write "billi").
- Do not repeat character traits inside each scene_visual_prompts item.
- Output JSON only, nothing else.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM output: {text[:300]}")
    return json.loads(text[start:end + 1])


# Defense-in-depth: even with explicit instructions, a small LLM sometimes still
# slips in a transliterated Hindi name or a second character (e.g. "shopkeeper").
# Both derail image generation (wrong/extra subjects), so strip them post-hoc.
_OTHER_CHARACTER_WORDS = [
    "shopkeeper", "shop owner", "owner", "vendor", "merchant",
    "man", "woman", "person", "people", "girl", "boy", "human",
]


def _sanitize_visual_prompt(vprompt: str, species: str, hindi_word: str) -> str:
    text = vprompt
    for bad in [hindi_word, hindi_word.lower(), hindi_word.capitalize()]:
        text = text.replace(bad, f"the {species}")
    for word in _OTHER_CHARACTER_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.")
    return text or f"the {species} in the story"


def _normalize_durations(scenes: list, target_duration: float):
    total = sum(float(s.get("duration", 0)) or 0 for s in scenes)
    if total <= 0:
        per = target_duration / len(scenes)
        for s in scenes:
            s["duration"] = round(per, 2)
        return
    scale = target_duration / total
    for s in scenes:
        s["duration"] = round(float(s["duration"]) * scale, 2)


def generate_scenes(story_text: str, num_scenes: int = 6, target_duration: float = 30.0) -> dict:
    """Returns dict with title, character_description, scenes[]. Raises on failure (no silent fallback)."""
    import torch

    narrations = split_story_into_scenes(story_text, num_scenes)
    n = len(narrations)

    model, tokenizer = load_llm()
    scene_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(narrations))
    species_match = detect_character_species(story_text)
    if species_match:
        hindi_word, species = species_match
        prompt = PROMPT_TEMPLATE_KNOWN_SPECIES.format(
            n=n, scene_list=scene_list, species=species, hindi_word=hindi_word,
        )
    else:
        prompt = PROMPT_TEMPLATE.format(n=n, scene_list=scene_list)
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=800,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    data = _extract_json(raw)

    visual_prompts = data.get("scene_visual_prompts") or []
    if len(visual_prompts) != n:
        # pad/truncate defensively rather than failing the whole run on a minor LLM miscount
        if len(visual_prompts) < n:
            visual_prompts = visual_prompts + ["a scene from the story"] * (n - len(visual_prompts))
        else:
            visual_prompts = visual_prompts[:n]

    if species_match:
        hindi_word, species = species_match
        extra_traits = (data.get("character_extra_traits") or "").strip()
        char_desc = f"{species}, {extra_traits}" if extra_traits else species
        visual_prompts = [_sanitize_visual_prompt(v, species, hindi_word) for v in visual_prompts]
        solo_tag = f"solo, single {species}, only one {species}, "
    else:
        char_desc = (data.get("character_description") or "").strip() or config.CHARACTER_DESCRIPTION
        solo_tag = "solo, single character, "

    per_scene_duration = target_duration / n
    scenes = []
    for i, (narration, vprompt) in enumerate(zip(narrations, visual_prompts), start=1):
        scenes.append({
            "scene": i,
            "duration": round(per_scene_duration, 2),
            "narration_hindi": narration,
            "visual_prompt": vprompt,
            "visual_prompt_full": f"{solo_tag}{vprompt}, {char_desc}, 2D cartoon style",
        })

    _normalize_durations(scenes, target_duration)

    return {
        "title": data.get("title", "Untitled"),
        "character_description": char_desc,
        "scenes": scenes,
    }
