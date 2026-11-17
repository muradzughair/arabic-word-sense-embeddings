"""
Helpers for extracting and comparing contextual embeddings of Arabic words.

The main idea: a polysemous word such as "هدف" (goal / objective) should get a
different vector depending on the sentence it appears in. To measure that, we
take the hidden states of the sub-word tokens that make up the word, average
them into one word vector, and compare those vectors with cosine similarity.
"""

import gc
import re
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class ModelSpec:
    """Everything we need to know to load and use one encoder."""

    label: str            # short name used in tables and plots
    hf_id: str            # Hugging Face model id
    architecture: str     # "encoder" (bidirectional) or "decoder" (causal)
    prefix: str = ""      # some models expect a prefix before the text (E5 uses "passage: ")
    large: bool = False   # large models are skipped unless explicitly enabled


# The four model families compared in this project.
MODELS = [
    ModelSpec("AraBERT v02", "aubmindlab/bert-base-arabertv02", "encoder"),
    ModelSpec("BGE-M3", "BAAI/bge-m3", "encoder"),
    ModelSpec("mE5-base", "intfloat/multilingual-e5-base", "encoder", prefix="passage: "),
    ModelSpec("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "decoder"),
    ModelSpec("Qwen2.5-7B", "Qwen/Qwen2.5-7B", "decoder", large=True),
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(spec: ModelSpec, device: torch.device | None = None):
    """Load a tokenizer and model in evaluation mode."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

    # Decoder models (Qwen) usually ship without a padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if spec.architecture == "decoder" and device.type == "cuda":
        # Half precision keeps the larger Qwen checkpoints inside GPU memory.
        model = AutoModel.from_pretrained(spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModel.from_pretrained(spec.hf_id).to(device)

    model.eval()
    return tokenizer, model


def free_memory() -> None:
    """Release memory after deleting a model, before loading the next one."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_word_span(text: str, word: str) -> tuple[int, int]:
    """Character span of the first whole-word occurrence of `word` in `text`."""
    match = re.search(rf"(?<!\S){re.escape(word)}(?!\S)", text)
    if match is None:
        raise ValueError(f"'{word}' was not found as a separate word in: {text}")
    return match.span()


@torch.no_grad()
def encode(texts: list[str], tokenizer, model, return_offsets: bool = False):
    """Tokenize a batch and run it through the model."""
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=return_offsets,
    )
    offsets = batch.pop("offset_mapping", None)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    hidden = model(**batch).last_hidden_state.float().cpu()
    return batch, hidden, offsets


def word_embedding(sentence: str, word: str, tokenizer, model, prefix: str = "") -> torch.Tensor:
    """
    Contextual vector for `word` inside `sentence`.

    A word is often split into several sub-word tokens, so we average the
    hidden states of every token whose character range overlaps the word.
    """
    text = prefix + sentence
    start, end = find_word_span(text, word)

    _, hidden, offsets = encode([text], tokenizer, model, return_offsets=True)
    token_positions = [
        i for i, (s, e) in enumerate(offsets[0].tolist())
        if e > s and s < end and e > start      # skip special tokens, keep overlapping ones
    ]
    if not token_positions:
        raise ValueError(f"No tokens matched '{word}' in: {text}")

    return hidden[0, token_positions].mean(dim=0)


def sentence_embedding(sentences: list[str], tokenizer, model, prefix: str = "") -> torch.Tensor:
    """Mean-pooled sentence vectors (padding tokens are ignored)."""
    batch, hidden, _ = encode([prefix + s for s in sentences], tokenizer, model)
    mask = batch["attention_mask"].cpu().unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def evaluate_polysemy(spec: ModelSpec, tokenizer, model, words: list[dict]) -> pd.DataFrame:
    """
    For every polysemous word we have three sentences:
      - anchor:         the word used in sense A
      - same_sense:     another sentence with sense A
      - different_sense: a sentence with sense B

    A model that really understands context should give
    sim(anchor, same_sense) > sim(anchor, different_sense).
    The gap between the two is reported as the "sense separation" score.
    """
    rows = []
    for item in words:
        word = item["word"]
        anchor = word_embedding(item["anchor"], word, tokenizer, model, spec.prefix)
        same = word_embedding(item["same_sense"], word, tokenizer, model, spec.prefix)
        diff = word_embedding(item["different_sense"], word, tokenizer, model, spec.prefix)

        sim_same = cosine(anchor, same)
        sim_diff = cosine(anchor, diff)
        rows.append({
            "model": spec.label,
            "word": word,
            "senses": f"{item['sense_a']} vs {item['sense_b']}",
            "sim_same_sense": round(sim_same, 4),
            "sim_different_sense": round(sim_diff, 4),
            "sense_separation": round(sim_same - sim_diff, 4),
            "correct": sim_same > sim_diff,
        })
    return pd.DataFrame(rows)


def tokenization_report(sentences: list[str], tokenizer, prefix: str = "") -> pd.DataFrame:
    """How many tokens each sentence becomes, and what those tokens look like."""
    rows = []
    for s in sentences:
        ids = tokenizer(prefix + s, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        rows.append({
            "sentence": s,
            "words": len(s.split()),
            "tokens": len(tokens),
            "tokens_per_word": round(len(tokens) / len(s.split()), 2),
            "token_list": " | ".join(tokens),
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per model: average similarities, separation and accuracy."""
    return (
        results.groupby("model", sort=False)
        .agg(
            avg_sim_same_sense=("sim_same_sense", "mean"),
            avg_sim_different_sense=("sim_different_sense", "mean"),
            avg_sense_separation=("sense_separation", "mean"),
            accuracy=("correct", "mean"),
        )
        .round(4)
        .sort_values("avg_sense_separation", ascending=False)
    )
