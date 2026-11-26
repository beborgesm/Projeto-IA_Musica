"""First-order Markov baseline on the piano tokenizer's vocabulary."""

from collections import Counter
from itertools import pairwise
import math

from .tokenizer import FORMAT_VERSION, TOKEN_IDS, VOCABULARY, token_sequences


MODEL_FORMAT_VERSION = 1
ALPHA = 0.1
BOS = TOKEN_IDS["BOS"]
EOS = TOKEN_IDS["EOS"]
TARGET_VOCABULARY_SIZE = len(VOCABULARY) - 1  # BOS is available only as context.


def _valid_token(token) -> bool:
    return type(token) is int and 0 <= token < len(VOCABULARY)


def _corpus_transitions(corpus: dict) -> Counter:
    counts = Counter()
    for _, tokens in token_sequences(corpus):
        counts.update(pairwise(tokens))
    return counts


def fit_markov(corpus: dict) -> dict:
    """Fit bigram counts on training songs, resetting the context at each BOS."""
    if corpus["split"] != "train":
        raise ValueError(f"Expected the train corpus, got {corpus['split']}")
    counts = _corpus_transitions(corpus)
    return {
        "model_format_version": MODEL_FORMAT_VERSION,
        "order": 1,
        "alpha": ALPHA,
        "token_format_version": FORMAT_VERSION,
        "vocabulary": VOCABULARY,
        "training_songs": len(corpus["songs"]),
        "training_transitions": counts.total(),
        "counts": [[context, target, count] for (context, target), count in sorted(counts.items())],
    }


def _model_counts(model: dict) -> tuple[Counter, Counter, float]:
    if (
        model["model_format_version"] != MODEL_FORMAT_VERSION
        or model["order"] != 1
        or model["token_format_version"] != FORMAT_VERSION
        or tuple(model["vocabulary"]) != VOCABULARY
    ):
        raise ValueError("Incompatible first-order Markov checkpoint")
    alpha = float(model["alpha"])
    if alpha <= 0 or not math.isfinite(alpha * TARGET_VOCABULARY_SIZE):
        raise ValueError("Expected a finite positive smoothing weight")

    counts = Counter()
    contexts = Counter()
    for context, target, count in model["counts"]:
        if (
            not _valid_token(context)
            or not _valid_token(target)
            or context == EOS
            or target == BOS
            or type(count) is not int
            or count <= 0
            or (context, target) in counts
        ):
            raise ValueError("Invalid transition counts in the checkpoint")
        counts[context, target] = count
        contexts[context] += count
    if not counts or counts.total() != model["training_transitions"]:
        raise ValueError("The checkpoint's training transition total is inconsistent")
    return counts, contexts, alpha


def next_token_log_probabilities(model: dict) -> list[list[float]]:
    """Smoothed log P(target | context) for every context ID, with the smoothing of evaluate_markov.

    Row ``context`` holds one value per output class, where class ``i`` is token ID ``i + 1``,
    because BOS is never a target.
    """
    counts, contexts, alpha = _model_counts(model)
    smoothing_mass = alpha * TARGET_VOCABULARY_SIZE
    rows = []
    for context in range(len(VOCABULARY)):
        denominator = math.log(contexts[context] + smoothing_mass)
        rows.append([math.log(counts[context, target] + alpha) - denominator for target in range(1, len(VOCABULARY))])
    return rows


def evaluate_markov(model: dict, corpus: dict) -> dict:
    """Measure token-weighted perplexity, including EOS predictions."""
    counts, contexts, alpha = _model_counts(model)
    observed = _corpus_transitions(corpus)
    smoothing_mass = alpha * TARGET_VOCABULARY_SIZE
    predicted = observed.total()
    mean_nll = math.fsum(
        frequency
        * (math.log(contexts[context] + smoothing_mass) - math.log(counts[context, target] + alpha))
        for (context, target), frequency in observed.items()
    ) / predicted
    return {
        "split": corpus["split"],
        "order": model["order"],
        "alpha": alpha,
        "songs": len(corpus["songs"]),
        "predicted_tokens": predicted,
        "mean_nll_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
    }
