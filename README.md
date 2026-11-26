# Projeto Final — IA Generativa para Música (DCC831, UFMG, 2026/2)

Symbolic piano music generation on POP909. We train our own decoder-only Transformer in PyTorch and an n-gram/Markov baseline on the same REMI tokens. Then we generate pop piano excerpts and evaluate them objectively against human test songs.

| Research question | Evaluation |
|---|---|
| RQ1: Does the neural model predict musical events better than a limited-context model? | Test perplexity on identical tokens |
| RQ2: Are its generations statistically closer to human music than the baseline's? | Symbolic metrics, OA and KL (Yang & Lerch, 2020), listening notes |
| RQ3: Do the models reproduce the dataset's traits without copying training songs? | Exact matches, shared n-grams, longest shared subsequence, transposition-invariant similarity |
| RQ4: How does sampling temperature trade off diversity, coherence and memorization? | RQ2 and RQ3 repeated across temperatures, no retraining |

## Repository

- [`music/`](music/): the generated music. It holds five 32-bar excerpts from the Transformer and five from the Markov baseline, each as MIDI and as rendered MP3 audio.
- [`generation/`](generation/): all the code, as a Python package. It covers data preparation, the REMI tokenizer, the Markov and n-gram baselines, the causal Transformer, sampling and evaluation (validity, perplexity, music metrics, memorization, rendering and tables).
  - [`generation/README.md`](generation/README.md): data, representation, models, sampling and validity.
  - [`generation/EVALUATION.md`](generation/EVALUATION.md): music metrics, memorization, the temperature study, rendering and tables.
- [`generation/manifests/`](generation/manifests/): the song-level split and the excluded songs.
- [`generation/results/`](generation/results/): result JSONs, tables, figures and the [listening notes](generation/results/listening.md). The figures and tables used in the report are in [`generation/results/report/`](generation/results/report/).
- [`generation/figures/`](generation/figures/): the Transformer's training and validation curves.
- The root `requirements.txt` includes `generation/requirements.txt`.

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On a CUDA machine, install the matching PyTorch build first, as described in [`generation/README.md`](generation/README.md#causal-transformer). Rendering also needs FluidSynth and FFmpeg on the system.

## POP909 dataset

Run these commands from this repository's root to fetch the dataset at the revision used for the split.

```bash
git clone https://github.com/music-x-lab/POP909-Dataset.git ../dados/POP909-Dataset
git -C ../dados/POP909-Dataset checkout --detach d83e6edba6872a704f5d3b8b32f5cb540088dae6
```

The [split manifest](generation/manifests/pop909-split.csv) lists the primary MIDI of the 855 songs kept after the meter filter: 680 for training, 89 for validation and 86 for testing. The [excluded songs](generation/manifests/pop909-excluded.csv) are listed with the reason. The filter, the split rule and the dataset statistics are described in [`generation/README.md`](generation/README.md#data-and-filtering).

## Reproducing the results

Run each step from the repository root. Each step's commands and outputs are documented where the step is described. Steps 1 to 5 write token corpora, checkpoints and sampling runs to `generation/tokens/`, `generation/checkpoints/` and `generation/outputs/`. These folders are not stored in the repository.

| Step | Command | Documented in |
|---|---|---|
| 1. Filter and split POP909 | `python -m generation split` | [Data and filtering](generation/README.md#data-and-filtering) |
| 2. Tokenize and verify the round trip | `python -m generation prepare` | [Reproduce the data milestone](generation/README.md#reproduce-the-data-milestone) |
| 3. Fit the Markov baseline and choose its order | `python -m generation train-markov`, `select-ngram` | [Markov baseline](generation/README.md#first-order-markov-baseline) |
| 4. Train the Transformer | `python -m generation.neural train` | [Causal Transformer](generation/README.md#causal-transformer) |
| 5. Sample both systems and cut the human references | `python -m generation.sampling`, `python -m generation reference` | [Sampling and reference excerpts](generation/README.md#sampling-and-reference-excerpts) |
| 6. Validity and test perplexity | `python -m generation.report_validity`, `report_perplexity` | [Validity report](generation/README.md#validity-report) |
| 7. Music metrics, OA and KL | `python -m generation.metrics` | [Evaluation](generation/EVALUATION.md#music-metrics-and-distribution-comparison-rq2) |
| 8. Memorization | `python -m generation.memorization` | [Evaluation](generation/EVALUATION.md#memorization-rq3) |
| 9. Temperature study | `python -m generation.metrics`, `memorization` on the `temp-*` runs | [Evaluation](generation/EVALUATION.md#temperature-rq4) |
| 10. Render audio | `python -m generation.render` | [Evaluation](generation/EVALUATION.md#rendering) |
| 11. Build the tables | `python -m generation.tables` | [Evaluation](generation/EVALUATION.md#tables) |
| 12. Build the report's figures and tables | `python -m generation.paper` | [Evaluation](generation/EVALUATION.md#report-figures-and-tables) |

## Contributions

All four members took part in every stage of the project:
- planning and scoping;
- data preparation and representation;
- implementation and training of the models;
- sampling;
- evaluation and listening;
- interpretation of the results;
- writing of the report.

The work was discussed and decided together, and each part was reviewed by the group.

Members: Felipe Melo, Bernardo Borges, Gabriel Alkmim and Bruno Lopes.

## AI usage

As the assignment requires, AI tools are declared here with what they were used for and links to the conversations. The group reviewed every result and every change to the code and the text, and answers for them.

| Tool | What it was used for | Conversation |
|---|---|---|
| Gemini | Planning, explaining concepts, discussing methodological decisions, code review and critical review of the writing | [link](https://share.gemini.google/QDtBtl7tjCyf) |
| ChatGPT | Planning, explaining concepts, discussing methodological decisions, code review and critical review of the writing | [link](https://chatgpt.com/share/6b079829-07e0-83e9-a0a2-c4b5552429bc) |

## License

Code is under the MIT license (`LICENSE`). Datasets and the soundfont keep their original licenses.
