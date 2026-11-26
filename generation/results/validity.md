| System | Bars | Sampling | Temperature | Seeds | Valid | Invalid | Incomplete | Empty | Collisions per excerpt (median) | Collisions per note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| markov | 16 | free | 1 | 100 | 0 (0%) | 99 (99%) | 1 (1%) | 0 (0%) | 2 | 0.286 |
| transformer | 16 | free | 1 | 100 | 39 (39%) | 48 (48%) | 13 (13%) | 0 (0%) | 18 | 0.075 |
| markov | 16 | syntax mask | 1 | 115 | 100 (87%) | 0 (0%) | 15 (13%) | 0 (0%) | 27 | 0.194 |
| transformer | 16 | syntax mask | 1 | 134 | 100 (75%) | 0 (0%) | 34 (25%) | 0 (0%) | 19 | 0.072 |
| markov | 32 | free | 1 | 100 | 0 (0%) | 99 (99%) | 1 (1%) | 0 (0%) | 2 | 0.286 |
| transformer | 32 | free | 1 | 100 | 16 (16%) | 68 (68%) | 16 (16%) | 0 (0%) | 29.5 | 0.076 |
| markov | 32 | syntax mask | 1 | 140 | 100 (71%) | 0 (0%) | 40 (29%) | 0 (0%) | 56.5 | 0.192 |
| transformer | 32 | syntax mask | 1 | 173 | 100 (58%) | 0 (0%) | 73 (42%) | 0 (0%) | 36 | 0.069 |

Grammar errors behind invalid excerpts:

- markov, 16 bars, free: Expected increasing positions inside a bar (91), Expected a bar, a position, or a pitch after a position (7), Expected a positive duration and a velocity after each pitch (1).
- transformer, 16 bars, free: Expected increasing positions inside a bar (30), Expected a positive duration and a velocity after each pitch (11), Expected a bar, a position, or a pitch after a position (7).
- markov, 32 bars, free: Expected increasing positions inside a bar (91), Expected a bar, a position, or a pitch after a position (7), Expected a positive duration and a velocity after each pitch (1).
- transformer, 32 bars, free: Expected increasing positions inside a bar (40), Expected a positive duration and a velocity after each pitch (19), Expected a bar, a position, or a pitch after a position (9).

Incomplete excerpts stopped by EOS / by the token limit:

- markov, 16 bars, free: 1 / 0.
- transformer, 16 bars, free: 13 / 0.
- markov, 16 bars, syntax mask: 15 / 0.
- transformer, 16 bars, syntax mask: 34 / 0.
- markov, 32 bars, free: 1 / 0.
- transformer, 32 bars, free: 16 / 0.
- markov, 32 bars, syntax mask: 40 / 0.
- transformer, 32 bars, syntax mask: 73 / 0.
