# Listening notes

The group listened to the ten rendered 32-bar excerpts in [`music/`](../../music/): five per system, chosen by the lowest valid seeds. Each Transformer excerpt was rated by one member with a short questionnaire. The Markov excerpts were assessed together. The notes below are the members' own answers, summarized and translated into English.

The questionnaire, one answer per question:
1. Does it sound like music?
2. Is it pleasant to listen to?
3. Could you follow the rhythm?
4. Did you notice repetition?
5. What sounds wrong?
6. Could it pass for a human composition?
7. Overall score, out of 5.

## Transformer (τ = 1.0, syntax mask)

Three of the five ratings below have no seed mapping because the files were not recorded with them. The other two are `s6` and `s2`.

| Excerpt | Music? | Pleasant? | Rhythm | Repetition | What sounds wrong | Human? | Score |
|---|---|---|---|---|---|---|---:|
| `s6` | Yes | Yes | Followed it | Not noticed | Some "lost" notes near the end | Maybe; the ending is suspicious | 4.0 |
| `s2` | Yes | Yes, more so as it goes on | Followed the rhythm and the progression | Not noticed | Starts somewhat messy, then settles and improves | Yes | 3.8 |
| A | Yes | Yes; it sounds like a normal, human-made song | Followed it (listener has musical experience) | Not noticed | Rarely, a few notes leave the pattern | Yes | 4.8 |
| B | Yes | Yes, and lively | Followed the rhythm and the progression | Not noticed | Some abrupt, overly long pauses; the ending is silent | Yes | 4.2 |
| C | At the start; sometimes out of tune | In parts | Followed it | Seems to repeat at the start | Some out-of-tune notes; the background piano is beautiful | Yes, though it played wrong notes | 3.9 |

Mean score: 4.1 (range 3.8–4.8). All five sound like music with a rhythm that is easy to follow. Four could pass for human compositions, and one might. The complaints are local:
- out-of-key or "lost" notes;
- abrupt pauses and a silent ending;
- a messy opening that settles.

## Markov (order 1, τ = 1.0, syntax mask)

All five excerpts were judged bad. They sound like random, widely spaced notes. None sounds like music, and all are unpleasant to hear.

## How this matches the metrics

- The Transformer's out-of-key notes match its weakest metrics (`results/tables/metrics.md`). At 32 bars it uses 10.9 pitch classes per excerpt against 8.5 for human excerpts, and its scale consistency is 0.94 against 0.98.
- The Markov model's random-sounding, spaced notes match three of its metrics:
  - all 12 pitch classes used almost uniformly (pitch-class entropy 3.56 bits, with log₂ 12 = 3.58);
  - a 67-semitone range against 48;
  - 15% empty beats against 1%.

These notes are exploratory. Each excerpt had a single listener who knows the project.
