"""Interpolated n-gram baseline on the REMI vocabulary (order selection for the Markov baseline).

An order-n model predicts each token from the previous n tokens of the same song. Contexts
shorter than n at the start of a song are padded with BOS. Each order interpolates with the
order below it, down to a uniform distribution over the 353 possible targets:

    P_k(w | h) = (c(h, w) + T(h) * P_{k-1}(w | h')) / (c(h) + T(h))

where h' drops the oldest token of h, c counts training occurrences and T(h) is the number of
distinct tokens seen after h. An unseen context falls back to the order below, so every target
keeps a positive probability, including transitions never seen in training.

Absolute discounting is available as a robustness check on the smoothing choice:

    P_k(w | h) = max(c(h, w) - D, 0) / c(h) + D * T(h) / c(h) * P_{k-1}(w | h')
"""

from collections import Counter, defaultdict
import math

from .tokenizer import FORMAT_VERSION, TOKEN_IDS, VOCABULARY, token_sequences


MODEL_FORMAT_VERSION = 1
SMOOTHINGS = ("witten-bell", "absolute")
BOS = TOKEN_IDS["BOS"]
EOS = TOKEN_IDS["EOS"]
TARGETS = len(VOCABULARY) - 1  # class i is token ID i + 1; BOS is only ever context
UNIFORM = 1 / TARGETS


def _events(tokens: list[int], order: int):
    """Yield (context of length ``order``, target) for every predicted token of one song."""
    padded = [BOS] * (order - 1) + tokens if order else tokens
    for index in range(max(order, 1), len(padded)):
        yield tuple(padded[index - order : index]), padded[index]


class NGramModel:
    """Counts for every order from 0 to ``order``; ``levels[k]`` maps a length-k context to its targets."""

    def __init__(
        self,
        order: int,
        levels: list[dict[tuple, Counter]],
        training_songs: int,
        smoothing: str = "witten-bell",
        discount: float = 0.75,
    ):
        if order < 0 or len(levels) != order + 1:
            raise ValueError("Expected one count table per order from 0 to the model order")
        if smoothing not in SMOOTHINGS or not 0 < discount < 1:
            raise ValueError(f"Use smoothing in {SMOOTHINGS} and 0 < discount < 1")
        self.order = order
        self.smoothing = smoothing
        self.discount = discount
        self.levels = levels
        self.training_songs = training_songs
        # (total count, distinct targets) per context, used by every probability lookup.
        self.totals = [
            {context: (sum(targets.values()), len(targets)) for context, targets in level.items()}
            for level in levels
        ]

    def truncated(self, order: int) -> "NGramModel":
        """The same counts used as a lower-order model."""
        if not 0 <= order <= self.order:
            raise ValueError(f"Order {order} is outside 0..{self.order}")
        return NGramModel(order, self.levels[: order + 1], self.training_songs, self.smoothing, self.discount)

    def with_smoothing(self, smoothing: str, discount: float = 0.75) -> "NGramModel":
        return NGramModel(self.order, self.levels, self.training_songs, smoothing, discount)

    def _weights(self, total: int, types: int) -> tuple[float, float]:
        """(divisor of a discounted count, weight of the lower order) for one context."""
        if self.smoothing == "witten-bell":
            return total + types, types / (total + types)
        return total, self.discount * types / total

    def _seen(self, count: int) -> float:
        return count if self.smoothing == "witten-bell" else max(count - self.discount, 0)

    def probability(self, context: tuple, target: int) -> float:
        """P(target | context), where context holds the last ``order`` tokens (BOS-padded)."""
        probability = UNIFORM
        for k in range(self.order + 1):
            history = context[len(context) - k :] if k else ()
            targets = self.levels[k].get(history)
            if targets is None:
                break  # a longer context containing an unseen suffix is unseen as well
            divisor, lower = self._weights(*self.totals[k][history])
            probability = self._seen(targets.get(target, 0)) / divisor + lower * probability
        return probability

    def distribution(self, history: list[int]) -> list[float]:
        """Next-token probabilities over the output classes, given a song's tokens so far."""
        if not history or history[0] != BOS:
            raise ValueError("A history starts with BOS")
        padded = [BOS] * self.order + history
        context = tuple(padded[len(padded) - self.order :]) if self.order else ()
        probabilities = [UNIFORM] * TARGETS
        for k in range(self.order + 1):
            suffix = context[len(context) - k :] if k else ()
            targets = self.levels[k].get(suffix)
            if targets is None:
                break
            divisor, lower = self._weights(*self.totals[k][suffix])
            probabilities = [value * lower for value in probabilities]
            for target, count in targets.items():
                probabilities[target - 1] += self._seen(count) / divisor
        return probabilities

    def to_json(self) -> dict:
        return {
            "model_format_version": MODEL_FORMAT_VERSION,
            "smoothing": self.smoothing,
            "discount": self.discount,
            "order": self.order,
            "token_format_version": FORMAT_VERSION,
            "vocabulary": VOCABULARY,
            "training_songs": self.training_songs,
            "levels": [
                [[*context, target, count] for context, targets in sorted(level.items()) for target, count in
                 sorted(targets.items())]
                for level in self.levels
            ],
        }

    @classmethod
    def from_json(cls, model: dict) -> "NGramModel":
        if (
            model["model_format_version"] != MODEL_FORMAT_VERSION
            or model["token_format_version"] != FORMAT_VERSION
            or tuple(model["vocabulary"]) != VOCABULARY
        ):
            raise ValueError("Incompatible n-gram checkpoint")
        levels = []
        for k, rows in enumerate(model["levels"]):
            level = defaultdict(Counter)
            for row in rows:
                *context, target, count = row
                if len(context) != k or not 1 <= target < len(VOCABULARY) or count <= 0:
                    raise ValueError(f"Invalid count row at order {k}")
                level[tuple(context)][target] = count
            levels.append(dict(level))
        return cls(model["order"], levels, model["training_songs"], model["smoothing"], model["discount"])


def fit_ngram(corpus: dict, order: int, smoothing: str = "witten-bell", discount: float = 0.75) -> NGramModel:
    """Count every order from 0 to ``order`` on the training songs."""
    if corpus["split"] != "train":
        raise ValueError(f"Expected the train corpus, got {corpus['split']}")
    if order < 0:
        raise ValueError("The order must be at least zero")
    levels = [defaultdict(Counter) for _ in range(order + 1)]
    songs = 0
    for _, tokens in token_sequences(corpus):
        songs += 1
        for context, target in _events(tokens, order):
            for k in range(order + 1):
                levels[k][context[order - k :] if k else ()][target] += 1
    return NGramModel(order, [dict(level) for level in levels], songs, smoothing, discount)


def evaluate_ngram(model: NGramModel, corpus: dict) -> dict:
    """Token-weighted perplexity over every predicted token (EOS included, BOS never a target)."""
    total_nll = 0.0
    predicted = 0
    seen_contexts = 0
    smallest = 1.0
    for _, tokens in token_sequences(corpus):
        for context, target in _events(tokens, model.order):
            probability = model.probability(context, target)
            smallest = min(smallest, probability)
            total_nll -= math.log(probability)
            predicted += 1
            seen_contexts += context in model.levels[model.order]
    mean_nll = total_nll / predicted
    return {
        "split": corpus["split"],
        "order": model.order,
        "smoothing": model.smoothing,
        "discount": model.discount if model.smoothing == "absolute" else None,
        "songs": len(corpus["songs"]),
        "predicted_tokens": predicted,
        "mean_nll_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
        "seen_context_rate": seen_contexts / predicted,
        "smallest_probability": smallest,
    }


def select_order(
    train_corpus: dict, val_corpus: dict, max_order: int, smoothings: list[tuple[str, float]] | None = None
) -> dict:
    """Fit once up to ``max_order`` and score every order and smoothing on validation.

    The selected order is the one with the lowest validation perplexity under Witten-Bell,
    the reference smoothing; the other smoothings show whether that choice depends on it.
    """
    if val_corpus["split"] != "val":
        raise ValueError(f"Expected the val corpus, got {val_corpus['split']}")
    smoothings = smoothings or [("witten-bell", 0.75)]
    full = fit_ngram(train_corpus, max_order)
    rows = []
    for smoothing, discount in smoothings:
        for order in range(1, max_order + 1):
            model = full.truncated(order).with_smoothing(smoothing, discount)
            rows.append({**evaluate_ngram(model, val_corpus), "contexts": len(model.levels[order])})
    reference = [row for row in rows if row["smoothing"] == smoothings[0][0]]
    best = min(reference, key=lambda row: row["mean_nll_nats"])
    return {"selected_order": best["order"], "training_songs": full.training_songs, "orders": rows}
