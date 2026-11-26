# Evaluation

These tools compare generated excerpts with human ones (RQ2, RQ3) and render them for listening. Each one takes either of these inputs:
- **A run folder**, as written by `generation.sampling` or `generation reference`. The tools read its `generations.csv` and use only the rows whose `status` is `valid`. Generated sequences come from the folder's `tokens.json`. Human rows are read back from the token corpus of their split (`generation/tokens/<split>.json`, set with `--corpora`). The bar count comes from the folder.
- **A token corpus JSON**, as written by `prepare`, with `--bars`. `--sample` draws a fixed random subset of it. This is how the human training line is built.

Every excerpt goes through `excerpts.classify`, the rule the validity report also uses (see [Sampling and reference excerpts](README.md#sampling-and-reference-excerpts)). On top of that, notes that ring past the last bar are clipped at its end, so all excerpts cover exactly N bars.

The final evaluation uses these run folders in `generation/outputs/`, written by the sampling and reference commands in the [README](README.md#sampling-and-reference-excerpts). They hold the syntax mask, temperature 1.0 and 100 valid excerpts per system:
- `eval-markov-b{16,32}` and `eval-transformer-b{16,32}`;
- the human references `reference-b{16,32}`, which are the opening bars of the 86 test songs.

The same commands apply to the temperature runs `temp-*`.

## Music metrics and distribution comparison (RQ2)

`generation/metrics.py` computes seven muspy metrics per excerpt, using the 12-step-per-quarter grid and 48-step bars:
- pitch range;
- number of pitch classes used;
- pitch-class entropy;
- scale consistency;
- polyphony;
- empty-beat rate;
- groove consistency.

The comparison follows Yang & Lerch (2020). For each metric, the intra-set distances of the reference set are the absolute differences between all pairs of reference excerpts. The inter-set distances are the differences between every evaluated excerpt and every reference excerpt. Both distance samples go through the same estimation procedure:
- a Gaussian kernel density with Scott's bandwidth;
- the same 1,000-point grid, from zero to the largest distance.

The tool then reports two numbers:
- **Overlap area (OA):** the integral of the smaller of the two densities.
- **KL divergence:** KL from the intra-set density to the inter-set density.

A floor of `1e-10` is added to both densities before the KL, so that regions where one density vanishes give a finite value. An evaluated set that resembles the reference gives an OA near 1 and a KL near 0. The report also gives each set's mean and standard deviation, and how many values are undefined.

```bash
O=generation/outputs R=generation/results
for bars in 16 32; do
    for system in markov transformer; do
        python -m generation.metrics --evaluated $O/eval-$system-b$bars --reference $O/reference-b$bars \
            --output $R/metrics-$system-b$bars.json --figure $R/metrics-$system-b$bars.png
    done
    # Human line: 100 training songs (seed 0) cut to the same bar count as the reference run.
    python -m generation.metrics --evaluated generation/tokens/train.json --sample 100 --seed 0 \
        --reference $O/reference-b$bars --output $R/metrics-train-b$bars.json --figure $R/metrics-train-b$bars.png
done
```

### Results

[`results/tables/metrics.md`](results/tables/metrics.md) (and `.tex`) has every metric, set and length. The figures `results/metrics-*-b{16,32}.png` show the value distributions against the test excerpts.

The human training line gives an OA of 0.87 to 0.98. It is an empirical reference rather than an upper bound, since the OA can reach 1. The metrics describe the valid excerpts only. The Transformer comes close on most metrics. It is weakest on pitch classes used (OA 0.57 at 16 bars, 0.72 at 32) and scale consistency (0.70, 0.62).

The Markov model matches polyphony and groove reasonably well, but misses tonality almost completely:
- pitch-class entropy: OA 0.02 at 16 bars, 0.07 at 32;
- scale consistency: OA 0.00 at both lengths;
- empty-beat rate at 32 bars: OA 0.00.

This fits a model that only sees the previous token and so has little means to keep a key.

## Memorization (RQ3)

`generation/memorization.py` checks how much of each excerpt also appears in the training songs. Pieces are compared as note sequences, ordered by onset and then pitch. Velocity is ignored.

A run of notes matches a run in a training song when:
- every note has the same duration;
- every note after the first has the same onset gap from its predecessor;
- in the **exact** view, every note has the same pitch;
- in the **transposition-invariant** view, every note after the first has the same pitch interval from its predecessor, so a copy in another key still matches.

The first note's gap and interval point outside the run, so they are never compared. Matching uses 64-bit polynomial hashes of every window.

For each view the tool reports:
- **Exact copies:** pieces whose whole excerpt appears contiguously in some training song.
- **Shared n-gram fraction:** the share of the piece's n-note windows, for n = 4, 8 and 16, that occur anywhere in training.
- **Longest shared run:** the longest contiguous run of notes found in training, as a note count and as a fraction of the piece.

The output JSON also has these values for every piece. Training songs are compared in full, not as excerpts.

```bash
for run in eval-markov-b16 eval-markov-b32 eval-transformer-b16 eval-transformer-b32 reference-b16 reference-b32; do
    python -m generation.memorization --train generation/tokens/train.json \
        --pieces generation/outputs/$run --output generation/results/memorization-$run.json
done
```

A run takes about 1.5 minutes on an M3 Max, mostly to decode and hash the training songs. Runs are independent and can go in parallel.

### Results

[`results/tables/memorization.md`](results/tables/memorization.md) (and `.tex`) has the full table. No whole excerpt, generated or human, was found in a training song, and no excerpt shares a 16-note run with training.

The test songs show how much two different human songs overlap: their longest shared run averages 3.6 to 4.7 notes, with a maximum of 12.
- **Transformer:** at that same level, with mean 3.7 to 4.6, maximum 7, and a similar share of 4-grams. By these measures, its overlap with training is comparable to the overlap between human songs.
- **Markov:** shares less than the humans (mean 3.0 to 3.5, maximum 5, and under a tenth of the human 4-gram rate). This reflects excerpts that are less like the data, not more original ones.

## Temperature (RQ4)

The temperature runs `temp-{markov,transformer}-b{16,32}-t{0.5,0.8,1.2,1.5}` use the same model, the syntax mask and the same stopping rule as the τ = 1 evaluation runs. Each one stops at 100 valid excerpts or 2,000 seeds. Validity is reported for every run. Metrics and memorization need at least 20 valid excerpts, so only the Transformer at 32 bars with τ = 1.5, which has 2 valid excerpts out of 2,000, is left out.

```bash
O=generation/outputs R=generation/results
for run in $O/temp-*; do
    name=$(basename $run); bars=${name#*-b}; bars=${bars%%-*}
    [ "$(grep -c ',valid,' $run/generations.csv)" -lt 20 ] && continue
    python -m generation.metrics --evaluated $run --reference $O/reference-b$bars --output $R/metrics-$name.json
    python -m generation.memorization --train generation/tokens/train.json --pieces $run \
        --output $R/memorization-$name.json
done
python -m generation.paper --out-dir generation/results/report
```

`generation.paper` collects validity, mean OA and memorization per temperature, with τ = 1 taken from the evaluation runs. It writes the trend figure [`results/report/fig_temperature.pdf`](results/report/fig_temperature.pdf) and the table `tab_temperature.tex`. [Report figures and tables](#report-figures-and-tables) describes its other outputs.

Results at 32 bars:
- **τ = 0.8:** the best Transformer setting. Mean OA is 0.90 against a human reference of 0.93, and 92% of seeds are valid.
- **τ = 0.5:** the Transformer becomes conservative. Its pitch range shrinks, and 6.7% of its 4-grams occur in training, eleven times the human rate. It still copies no run of 16 notes, and its longest shared run is 9 notes at most.
- **τ ≥ 1.2:** early `EOS` dominates. Only 4% of seeds reach 32 bars at τ = 1.2, and 2 of 2,000 at τ = 1.5.
- **Markov:** its mean OA stays between 0.32 and 0.40 at every temperature.

## Rendering

`generation/render.py` turns excerpts into MIDI and MP3 files. Human and generated pieces go through the same steps:
- **Soundfont:** [MuseScore General](https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/) v0.2 (`MuseScore_General.sf3`, MIT license). It is pinned by SHA-256, so another file is rejected.
- **Synthesis:** FluidSynth at 44.1 kHz with its default settings (program 0, Grand Piano). The rendered audio runs about 3 s past the last note-off, so the final release is kept.
- **Loudness and encoding:** FFmpeg's single-pass `loudnorm` (EBU R128) targets −16 LUFS integrated with −1.5 dBTP, then the audio is encoded as a VBR MP3 (`-q:a 2`). Single-pass normalization lands within about ±1.5 LU of the target. Five validation excerpts measured between −17.2 and −15.3 LUFS.

Each excerpt MIDI starts at tick zero at its first bar. It keeps the tempo in effect there and the tempo changes inside the excerpt.

Pieces are chosen by a fixed rule, not by ear: the valid rows of a run with the lowest seeds. Human rows, which have no seed, are taken in ID order. The tool copies each MIDI from the run folder and renders it.

Install the tools first, for example with `brew install fluid-synth ffmpeg` or `apt install fluidsynth ffmpeg`. Then fetch the soundfont outside the repository and render:

```bash
mkdir -p ../dados/soundfonts
curl -fsSL -o ../dados/soundfonts/MuseScore_General.sf3 \
    https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3
for system in markov transformer; do
    python -m generation.render generation/outputs/eval-$system-b32 --count 5 \
        --soundfont ../dados/soundfonts/MuseScore_General.sf3 --out-dir music/$system
done
```

The deliverable music is in [`music/`](../music/): five 32-bar excerpts per system, with seeds 0, 1, 2, 3 and 6 for Markov and 1, 2, 3, 5 and 6 for the Transformer. Each comes as MIDI and MP3, from 72 to 120 s long.

Human POP909 excerpts can be rendered the same way to compare by ear. They are not included in the repository.

## Tables

`generation/tables.py` turns result JSONs into Markdown and LaTeX (booktabs) tables in `generation/results/tables/`. Result tables come only from this script. The current tables hold the final evaluation at temperature 1.0:

```bash
R=generation/results
python -m generation.tables \
    --metrics $R/metrics-train-b16.json $R/metrics-markov-b16.json $R/metrics-transformer-b16.json \
              $R/metrics-train-b32.json $R/metrics-markov-b32.json $R/metrics-transformer-b32.json \
    --metrics-labels "Human train, 16" "Markov, 16" "Transformer, 16" "Human train, 32" "Markov, 32" "Transformer, 32" \
    --memorization $R/memorization-reference-b16.json $R/memorization-eval-markov-b16.json \
                   $R/memorization-eval-transformer-b16.json $R/memorization-reference-b32.json \
                   $R/memorization-eval-markov-b32.json $R/memorization-eval-transformer-b32.json \
    --memorization-labels "Human test, 16" "Markov, 16" "Transformer, 16" "Human test, 32" "Markov, 32" "Transformer, 32" \
    --out-dir $R/tables
```

For the temperature study, add one result file per temperature in the order the table should show them.

## Report figures and tables

`generation/paper.py` writes the figures and compact tables used in the report, with Portuguese labels, to [`results/report/`](results/report/):
- the dataset table;
- the perplexity and validity table;
- the metrics and memorization tables at 32 bars;
- the temperature table;
- the training curves, from the history stored in the checkpoint;
- the metric densities;
- the temperature trend.

It reads the run folders, the result JSONs and `generation/checkpoints/transformer-remi.pt`.

```bash
python -m generation.paper --out-dir generation/results/report
```
