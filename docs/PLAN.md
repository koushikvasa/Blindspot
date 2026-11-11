# Blindspot: modules and build plan

## Data flow
```
real survey files ─► data/loaders ─► Respondents (people × items, demographics)
                                         │
                                  eval/splits (pool / test / holdout)
                                         │
generators (Claude personas, Flashpoint panel, baselines) ─► synthetic Respondents
                                         │
                 space (encode items, Gower distance, 2-D map)
                                         │
             coverage (nearest-synthetic distance per real person,
                       normalized by human-human distance → uncovered set,
                       precision / coverage metrics)
                                         │
agent loop:  cluster uncovered → describe as answer patterns (Claude)
          → request examples from pool (later: Flashpoint real respondents)
          → rewrite persona recipes → regenerate → re-measure on test
                                         │
                  api (FastAPI) ─► web (Constellation UI, live judge mode)
```

## Modules
| Module | Responsibility | Handover |
| --- | --- | --- |
| `blindspot/data` | Load Twin-2K-500 and GSS into one `Respondents` format; pick 20–30 items in themes | H1 |
| `blindspot/eval/splits.py` | Seeded pool/test/holdout split, saved to disk | H1 |
| `blindspot/space` | Item encoding, Gower distance, nearest neighbours, UMAP map coords | H2 |
| `blindspot/coverage` | Uncovered detection, coverage score, precision/coverage metrics, human-human scale | H2 |
| `blindspot/generators/baselines.py` | Independent marginals, Gaussian copula, add-N-real | H2 |
| `blindspot/generators/claude_personas.py` | Persona-recipe generator via Claude (cached, batched) | H3 |
| `blindspot/agent` | Gap clustering, gap description, request, recipe learning, round loop + logs | H4 |
| `blindspot/generators/flashpoint.py` | Audit Flashpoint synthetic panel; optional real-respondent request | H5 |
| `blindspot/api` | Endpoints for map, rounds, gaps, metrics, live judge submit | H6 |
| `web/` | Constellation UI (after design mockup is approved) | H7 |
| `blindspot/eval/final.py` | Headline numbers on holdout, results table for README | H8 |

## Milestones
- **Wed:** H1 done (data loads, splits saved, items chosen).
- **Thu:** H2–H4 (first coverage numbers vs baselines, then the agent loop working end to end).
- **Fri:** H5–H8 (Flashpoint, API, UI, final numbers), rehearsal. Stop by evening.
- **Sat:** adapt to organizer announcements, live mode, pitch.
