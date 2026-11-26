| System | Bars | Sampling | Temperature | Seeds | Valid | Invalid | Incomplete | Empty | Collisions per excerpt (median) | Collisions per note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| markov | 16 | syntax mask | 0.5 | 100 | 100 (100%) | 0 (0%) | 0 (0%) | 0 (0%) | 6 | 0.072 |
| transformer | 16 | syntax mask | 0.5 | 111 | 100 (90%) | 0 (0%) | 9 (8%) | 2 (2%) | 7 | 0.030 |
| markov | 16 | syntax mask | 0.8 | 102 | 100 (98%) | 0 (0%) | 2 (2%) | 0 (0%) | 18.5 | 0.151 |
| transformer | 16 | syntax mask | 0.8 | 107 | 100 (93%) | 0 (0%) | 7 (7%) | 0 (0%) | 17 | 0.055 |
| markov | 16 | syntax mask | 1 | 115 | 100 (87%) | 0 (0%) | 15 (13%) | 0 (0%) | 27 | 0.194 |
| transformer | 16 | syntax mask | 1 | 134 | 100 (75%) | 0 (0%) | 34 (25%) | 0 (0%) | 19 | 0.072 |
| markov | 16 | syntax mask | 1.2 | 147 | 100 (68%) | 0 (0%) | 47 (32%) | 0 (0%) | 35 | 0.221 |
| transformer | 16 | syntax mask | 1.2 | 362 | 100 (28%) | 0 (0%) | 262 (72%) | 0 (0%) | 12 | 0.096 |
| markov | 16 | syntax mask | 1.5 | 197 | 100 (51%) | 0 (0%) | 97 (49%) | 0 (0%) | 37 | 0.246 |
| transformer | 16 | syntax mask | 1.5 | 2000 | 24 (1%) | 0 (0%) | 1976 (99%) | 0 (0%) | 3 | 0.099 |
| markov | 32 | syntax mask | 0.5 | 100 | 100 (100%) | 0 (0%) | 0 (0%) | 0 (0%) | 14 | 0.076 |
| transformer | 32 | syntax mask | 0.5 | 111 | 100 (90%) | 0 (0%) | 9 (8%) | 2 (2%) | 15 | 0.027 |
| markov | 32 | syntax mask | 0.8 | 107 | 100 (93%) | 0 (0%) | 7 (7%) | 0 (0%) | 40 | 0.149 |
| transformer | 32 | syntax mask | 0.8 | 109 | 100 (92%) | 0 (0%) | 9 (8%) | 0 (0%) | 32 | 0.051 |
| markov | 32 | syntax mask | 1 | 140 | 100 (71%) | 0 (0%) | 40 (29%) | 0 (0%) | 56.5 | 0.192 |
| transformer | 32 | syntax mask | 1 | 173 | 100 (58%) | 0 (0%) | 73 (42%) | 0 (0%) | 36 | 0.069 |
| markov | 32 | syntax mask | 1.2 | 209 | 100 (48%) | 0 (0%) | 109 (52%) | 0 (0%) | 61 | 0.222 |
| transformer | 32 | syntax mask | 1.2 | 2000 | 90 (4%) | 0 (0%) | 1910 (96%) | 0 (0%) | 12 | 0.095 |
| markov | 32 | syntax mask | 1.5 | 523 | 100 (19%) | 0 (0%) | 423 (81%) | 0 (0%) | 48 | 0.251 |
| transformer | 32 | syntax mask | 1.5 | 2000 | 2 (0%) | 0 (0%) | 1998 (100%) | 0 (0%) | 3 | 0.099 |

Grammar errors behind invalid excerpts:


Incomplete excerpts stopped by EOS / by the token limit:

- transformer, 16 bars, syntax mask: 0 / 9.
- markov, 16 bars, syntax mask: 2 / 0.
- transformer, 16 bars, syntax mask: 7 / 0.
- markov, 16 bars, syntax mask: 15 / 0.
- transformer, 16 bars, syntax mask: 34 / 0.
- markov, 16 bars, syntax mask: 47 / 0.
- transformer, 16 bars, syntax mask: 262 / 0.
- markov, 16 bars, syntax mask: 97 / 0.
- transformer, 16 bars, syntax mask: 1976 / 0.
- transformer, 32 bars, syntax mask: 0 / 9.
- markov, 32 bars, syntax mask: 7 / 0.
- transformer, 32 bars, syntax mask: 9 / 0.
- markov, 32 bars, syntax mask: 40 / 0.
- transformer, 32 bars, syntax mask: 73 / 0.
- markov, 32 bars, syntax mask: 109 / 0.
- transformer, 32 bars, syntax mask: 1910 / 0.
- markov, 32 bars, syntax mask: 423 / 0.
- transformer, 32 bars, syntax mask: 1998 / 0.
