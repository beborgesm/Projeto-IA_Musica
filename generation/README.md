# Symbolic music generation on POP909

This folder holds the project's code: data preparation, tokenization, the Markov baseline, the causal Transformer, sampling and the validity and perplexity reports. Music metrics, memorization, the temperature study, rendering and tables are documented in [`EVALUATION.md`](EVALUATION.md).

## Data and filtering

The corpus uses the primary MIDI of each POP909 song at the [documented revision](../README.md#pop909-dataset). Each song's `MELODY`, `BRIDGE` and `PIANO` tracks merge into a single polyphonic piano.

MIDI time signatures in POP909 are unreliable. For example, some songs carry `1/4` events although their bars clearly last longer. The filter therefore relies on the dataset's annotations instead:
- `index.xlsx` gives the beats per measure and quavers per beat. Only songs annotated `2 × 2` are kept, because simple duple meter is compatible with a 4/4 grid. This removes 28 songs in `3 × 2` (3/4), 23 in `2 × 3` (6/8) and 1 in `3 × 3` (9/8).
- `beat_midi.txt` gives one line per beat. In every kept song the beats fall exactly one quarter note apart in MIDI ticks. Two songs, `418` and `519`, are dropped because their beats drift by more than one grid step (0.174 and 0.085 quarters).

The split is by song. The 909 sorted song IDs are shuffled with Python's `random.Random(42)`. The first 727 go to training, the next 91 to validation and the last 91 to testing. Excluded songs are removed only after this assignment, so the filter cannot move a song between splits. A SHA-256 check over all 2,898 files, including 1,989 alternate arrangements, found no exact copy across splits.

- [Split manifest](manifests/pop909-split.csv): 855 songs. Read `song_id` as text to keep the leading zeros, and resolve `midi_path` relative to `POP909/`.
- [Excluded songs](manifests/pop909-excluded.csv): 54 songs, each with its original split and the reason. 47 were in training, 2 in validation and 5 in testing.

### Dataset statistics

Durations run from tick zero to the last note-off of the reconstructed MIDI, under the song's tempo map. Keys are the first entry of `key_audio.txt`. Every split contains all 24 keys. Notes are counted after merging and quantization.

| Split | Songs | Hours | Mean duration (s) | Mean notes | Major / minor | Mean bars |
|---|---:|---:|---:|---:|---:|---:|
| train | 680 | 47.17 | 249.7 | 1,650.7 | 381 / 299 | 82.7 |
| val | 89 | 6.15 | 248.9 | 1,638.7 | 53 / 36 | 82.4 |
| test | 86 | 5.82 | 243.7 | 1,598.7 | 48 / 38 | 81.3 |

## Representation (REMI)

The tokenizer uses REMI events. Explicit bars and positions inside the bar let us cut and compare excerpts by bar count, as the evaluation protocol requires. They also give the model a direct sense of metrical position, which a pure time-shift representation leaves implicit.

Each quarter note is divided into 12 steps, so the grid holds both sixteenth notes and eighth-note triplets. A bar is 48 steps, which is four quarter notes. The grid is anchored at the first downbeat in `beat_midi.txt` and moved back by whole bars so that the first bar holds the first note. This puts every annotated beat at position 0, 12, 24 or 36. On the training split, the mean onset rounding error is 0.022 steps, against 0.233 steps for a grid anchored at tick zero.

The bar grid is fixed. POP909's four-beat downbeat annotations sometimes insert two-beat bars, and after such an insertion the annotated downbeat falls mid-bar on our grid.

Token types:
- `Bar` opens the next bar.
- `Position_0` to `Position_47` give an onset within the bar.
- Each note then uses `Pitch`, one or more `Duration` tokens, and `Velocity`. `Duration` comes in chunks of up to 48 steps, so long notes fit the vocabulary.

Simultaneous notes are ordered by pitch, and empty bars appear as consecutive `Bar` tokens. For example, C4 and E4 struck together on the second beat of the first bar and held for half a quarter note produce this sequence:

```text
BOS Bar Position_12 Pitch_60 Duration_6 Velocity_80 Pitch_64 Duration_6 Velocity_72 EOS
```

Quantization follows these rules:
- Note starts and ends round to the nearest step, with halfway points rounding up.
- A note that rounds to zero length lasts one step.
- Notes of the same pitch that round to the same onset, including copies across the merged tracks, become one note with the latest end and the highest velocity.
- A note that overlaps a later note of the same pitch ends at that note's onset.

The sequence keeps the file's tick resolution, its tempo events at their original ticks and the grid origin. Decoding therefore restores the notes at their original times. The reconstructed MIDI has one program-0 track named `PIANO` and a conductor track for tempo. Sustain pedal, the track of origin and time signatures are outside the representation.

## Reproduce the data milestone

Run from the repository root with Python 3.11 and the dataset at the documented revision.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r generation/requirements.txt

python -m generation split \
    --dataset-root ../dados/POP909-Dataset/POP909 \
    --manifest generation/manifests/pop909-split.csv \
    --excluded generation/manifests/pop909-excluded.csv
for split in train val test; do
    python -m generation prepare \
        --dataset-root ../dados/POP909-Dataset/POP909 \
        --manifest generation/manifests/pop909-split.csv \
        --split "$split" --output "generation/tokens/$split.json"
done
```

For each song, `prepare` serializes and reopens the reconstructed MIDI. It then checks that the notes, tick resolution, tempo map and program-0 piano match the normalized input exactly.

The output JSON holds the vocabulary, the split statistics and one record per song under `songs`. Each record contains:
- the integer token IDs;
- the key and duration;
- the tick resolution and the grid origin;
- the tempo events, as `[tick, microseconds_per_quarter]` pairs.

Token counts include `BOS` and `EOS`. The rounding errors are mean absolute errors in grid steps, measured before overlap correction. The test split is tokenized only so its round trip can be verified and its songs used as the human reference. It must not be used for model selection.

Use `encode` and `decode` to inspect one song. `encode` anchors the grid with the `beat_midi.txt` next to the MIDI when there is one, and uses tick zero otherwise.

```bash
python -m generation encode ../dados/POP909-Dataset/POP909/062/062.mid generation/tokens/062.json
python -m generation decode generation/tokens/062.json generation/tokens/062-roundtrip.mid
```

## Tokenization results

All 855 songs passed the MIDI round trip. The vocabulary has 354 tokens. The bar columns give how many bars a context of 256, 512 or 1,024 tokens covers on average, computed as tokens per bar across the split.

| Split | Token IDs observed | Mean tokens per song | Tokens per bar | Bars in 256 | Bars in 512 | Bars in 1,024 |
|---|---:|---:|---:|---:|---:|---:|
| train | 308 | 5,862.9 | 70.9 | 3.6 | 7.2 | 14.4 |
| val | 298 | 5,826.4 | 70.7 | 3.6 | 7.2 | 14.5 |
| test | 299 | 5,674.4 | 69.8 | 3.7 | 7.3 | 14.7 |

A 16-bar excerpt therefore takes about 1,130 tokens, and a 32-bar excerpt about 2,270.

Merging and quantization on the training split:
- 33,596 same-onset duplicates were merged;
- 45,862 overlapping notes were shortened;
- 2,199 zero-length notes were extended to one step.

## First-order Markov baseline

The baseline counts next-token transitions within each training song. It uses additive smoothing with `alpha = 0.1` over the 353 possible targets. `BOS` supplies the initial context and `EOS` is a prediction target. The validation perplexity is the exponential of the mean negative log probability across predicted tokens.

With the environment active and the prepared split files available, run from the repository root.

```bash
python -m generation train-markov \
    --corpus generation/tokens/train.json \
    --output generation/checkpoints/markov-order1.json
python -m generation evaluate-markov \
    --model generation/checkpoints/markov-order1.json \
    --corpus generation/tokens/val.json \
    --output generation/checkpoints/markov-order1-val.json
```

The checkpoint was fitted on 680 songs and 3,986,114 next-token transitions. On the 89 validation songs, it scored 518,458 targets with a mean negative log probability of 3.56726 nats and a perplexity of **35.4194**. The test perplexity is reported [below](#test-perplexity-rq1).

### Choosing the order on validation

`generation/ngram.py` fits an interpolated n-gram of any order on the same tokens. A context shorter than the order at the start of a song is padded with `BOS`. Each order interpolates with the order below it, down to a uniform distribution over the 353 targets, so no target ever gets zero probability. The reference smoothing is Witten-Bell. Absolute discounting with three discounts checks whether the choice depends on the smoothing. One fit up to the maximum order serves every lower order, because an order-k model uses the same counts truncated to k.

```bash
python -m generation select-ngram \
    --train generation/tokens/train.json \
    --val generation/tokens/val.json \
    --max-order 6 --output generation/checkpoints/ngram-order-selection.json
```

Validation perplexity on the same 518,458 targets. The seen-context column is the share of validation targets whose full context occurs in training.

| Order | Contexts | Seen context | Witten-Bell | Absolute, D = 0.5 | Absolute, D = 0.75 | Absolute, D = 0.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 307 | 100.0% | **35.50** | **35.46** | **35.49** | **35.51** |
| 2 | 21,662 | 99.9% | 37.17 | 38.42 | 37.79 | 37.67 |
| 3 | 405,876 | 94.3% | 45.90 | 53.87 | 46.61 | 44.46 |
| 4 | 1,615,609 | 53.9% | 60.72 | 77.14 | 57.81 | 52.23 |
| 5 | 2,369,567 | 14.1% | 66.98 | 86.77 | 62.11 | 55.05 |
| 6 | 2,603,830 | 2.4% | 68.16 | 88.58 | 62.90 | 55.55 |

Order 1 is best under every smoothing, and each longer context does worse. Longer contexts do help on the training songs: on a subset of 60 training songs, order 2 scores 7% fewer nats than order 1. The extra context therefore overfits. Per token type, order 2 is worse on validation for `Pitch`, `Velocity`, `Duration` and `Position`. Our reading is that the previous token is often a song-specific value, such as a velocity, so conditioning on it splits the counts without predicting the next token better. The baseline stays the first-order model with additive smoothing above, whose validation perplexity of 35.42 is also the lowest in this comparison.

## Causal Transformer

The default model uses six causal attention layers, eight heads, 256 hidden units, and a context of 512 tokens. Training uses AdamW, dropout of 0.1, gradient clipping, and seed 42. The CLI uses CUDA for training by default. Automatic mixed precision uses BFloat16 on GPUs with native support and uses Float32 otherwise.

On the GPU host, create the Python 3.11 environment and install the PyTorch build that matches its NVIDIA driver. The example uses the official CUDA 12.6 wheel. [PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v291)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r generation/requirements.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Use the train and validation JSONs from the data milestone. Each song is split into consecutive blocks. Every next-token target is scored exactly once, including `EOS`. Padding is ignored and the context resets at song and block boundaries. This yields the same 518,458 validation targets as the Markov baseline.

```bash
python -m generation.neural train \
    --train generation/tokens/train.json \
    --val generation/tokens/val.json \
    --output generation/checkpoints/transformer-remi.pt \
    --device cuda --epochs 20 --batch-size 16
```

The run writes a checkpoint and `transformer-remi.csv` after every completed epoch. The checkpoint stores the latest model and optimizer for resuming, plus the best weights selected by validation loss. To resume, repeat the train command with `--resume` and the same data and settings. The `--epochs` flag sets the total number of epochs. Adjust `--batch-size`, `--context`, `--width`, `--heads`, and `--layers` when starting a new run to suit the available GPU.

Evaluate the best checkpoint on validation and plot learning curves from the CSV. The training curve records loss during optimizer updates. The curves of the reported run are in [`figures/transformer-remi-curves.png`](figures/transformer-remi-curves.png).

```bash
python -m generation.neural evaluate \
    --model generation/checkpoints/transformer-remi.pt \
    --corpus generation/tokens/val.json --device cuda
python -m generation.plot_training generation/checkpoints/transformer-remi.csv \
    generation/figures/transformer-remi-curves.png
```

## Test perplexity (RQ1)

Both final models were frozen before the test split was scored: the order-1 Markov model and the Transformer checkpoint `transformer-remi.pt` (context 512, best validation epoch 14). Both predict every token after `BOS`, `EOS` included, with no temperature, top-k or top-p. The Transformer runs in Float32. The script stops if the two systems score different token counts.

```bash
python -m generation.report_perplexity \
    --markov generation/checkpoints/markov-order1.json \
    --transformer generation/checkpoints/transformer-remi.pt \
    --corpus generation/tokens/val.json generation/tokens/test.json \
    --output generation/results --device cuda
```

The script writes [`results/perplexity.md`](results/perplexity.md):

| System | Split | Songs | Predicted tokens | Mean NLL (nats) | Perplexity |
|---|---|---:|---:|---:|---:|
| Markov (order 1) | val | 89 | 518,458 | 3.56726 | 35.4194 |
| Transformer | val | 89 | 518,458 | 2.13972 | 8.4970 |
| Markov (order 1) | test | 86 | 487,909 | 3.58306 | 35.9835 |
| Transformer | test | 86 | 487,909 | 2.13660 | 8.4706 |

The Transformer's test perplexity is about a quarter of the Markov model's. Test and validation agree for both systems, so model selection on validation did not overfit it.

## Sampling and reference excerpts

Every comparison uses excerpts with the same number of bars: 32 in the final evaluation, with 16 as a check. `generation/excerpts.py` applies one set of rules to human and generated sequences:
- An excerpt holds the first N bars of a sequence. Trailing empty bars are dropped before decoding.
- The token grammar is checked strictly. A position that does not increase inside its bar, a pitch without a position, or a pitch without a duration and a velocity makes the excerpt `invalid`.
- Notes of one pitch that collide follow the tokenizer's normalization rule, which already shaped the human data. Notes on the same onset merge, and a note still sounding when its pitch starts again ends at that onset. The table counts both repairs per excerpt.
- An excerpt is `incomplete` when the sequence ends before N bars, by `EOS` or by the token limit, and `empty` when it holds no note.

The human reference is the opening N bars of each of the 86 test songs, matching unconditional generation, which also starts at the beginning of a piece. The excerpts keep each song's tempo map and grid origin.

```bash
for bars in 16 32; do
    python -m generation reference --corpus generation/tokens/test.json \
        --bars "$bars" --output "generation/outputs/reference-b$bars"
done
```

`generation/sampling.py` samples both systems with one sampler. Each system supplies next-token scores, and the sampler applies temperature, then top-k, then top-p. A piece stops at `EOS`, at the `Bar` that would open bar N + 1, or after 200 tokens per bar, about three times the human density. Every piece draws from its own generator seeded with its seed, so a piece depends only on its seed, the model and the settings. Each generated excerpt takes the initial tempo of a training song drawn with its seed.

Sampling is free by default. With `--syntax`, a mask identical for both systems allows only tokens the decoder accepts at that point. Collisions are not masked. With `--until-valid`, seeds are drawn in order until `--count` excerpts are valid. Every drawn seed stays in the table, so the same file gives the validity rate, and the evaluation set is the valid rows.

```bash
for bars in 16 32; do
    python -m generation.sampling markov --model generation/checkpoints/markov-order1.json \
        --train generation/tokens/train.json --output "generation/outputs/eval-markov-b$bars" \
        --bars "$bars" --syntax --until-valid --count 100
    python -m generation.sampling transformer --model generation/checkpoints/transformer-remi.pt \
        --train generation/tokens/train.json --output "generation/outputs/eval-transformer-b$bars" \
        --bars "$bars" --syntax --until-valid --count 100 --device cuda
done
```

Drop `--syntax --until-valid` for the free-sampling validity runs. Each output folder holds `generations.csv`, the decodable excerpts under `midi/` and the raw sampled sequences in `tokens.json`.

### Validity at temperature 1.0

Temperature 1.0, no top-k and no top-p. Under free sampling, the order-1 Markov model breaks the order of positions in almost every excerpt, because the previous token alone does not reveal how far into the bar it is. The comparisons in RQ2–RQ4 therefore use the syntax mask for both systems, and free sampling is reported as a validity result.

| System | Bars | Free: valid / invalid / incomplete (100 seeds) | Masked: seeds for 100 valid |
|---|---:|---|---:|
| Markov | 16 | 0 / 99 / 1 | 115 |
| Transformer | 16 | 39 / 48 / 13 | 134 |
| Markov | 32 | 0 / 99 / 1 | 140 |
| Transformer | 32 | 16 / 68 / 16 | 173 |

Under the mask, every non-valid excerpt is `incomplete` and ends with an early `EOS`. In the training data, `EOS` appears only at the end of songs about 82 bars long.

### Validity report

```bash
python -m generation.report_validity generation/outputs/free-* generation/outputs/eval-* \
    --output generation/results
```

The script writes [`results/validity.md`](results/validity.md). It reports the four statuses per run, the grammar errors behind invalid excerpts, whether incomplete excerpts stopped at `EOS` or at the token limit, and the same-pitch collisions that normalization resolved in decodable excerpts. The human excerpts have no collisions, because the tokenizer already normalized them. Every incomplete excerpt in these runs stopped at `EOS`, and none reached the token limit. Collisions per note are about 0.07 for the Transformer and 0.19 for the Markov model.
