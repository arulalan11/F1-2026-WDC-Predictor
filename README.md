<div align="center">

# 🏎️ F1 2026 WORLD CHAMPIONSHIP PREDICTOR 🤖

*Built to answer one question — can you predict who wins an F1 title from data alone, when the car matters almost as much as the driver?*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

![Domain](https://img.shields.io/badge/Domain-F1_Motorsport_Analytics-orange?style=flat-square)
![Model](https://img.shields.io/badge/Model-Random_Forest_Classifier-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Season--Level_ML_Classification-brightgreen?style=flat-square)

</div>

---

## Overview

This project treats WDC prediction as a **season-level classification problem**: given where a driver stands at any point in a season (points, recent form, car strength, gap to the leader), how likely are they to end up champion? Rather than a single black-box number, the model is validated season-by-season and explained through feature importance — because in F1, being able to say *why* a prediction was made matters as much as the prediction itself.

**Current model output (as of Round 12, 2026 season):**

| Driver | Team | Points | Title Chance |
|---|---|---|---|
| Antonelli | Mercedes | 242 | 83.9% |
| Russell | Mercedes | 183 | 14.5% |
| Hamilton | Ferrari | 183 | 1.1% |
| Norris | McLaren | 159 | 0.3% |
| Leclerc | Ferrari | 155 | 0.1% |

## Why this isn't just "points → winner"

F1 is a team sport wearing an individual sport's scoreboard. A driver's result is a mix of car performance and personal skill, so this project deliberately builds features that try to separate the two — most importantly, comparing each driver's qualifying pace **to their own teammate** (same car, so the gap is actually about the driver). The model's own feature importance output confirms what most F1 fans already suspect: constructor/car strength is one of the strongest predictors of who wins the title.

## Dataset

Sourced from the [Formula 1 Race Data](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) Kaggle dataset (Ergast-schema), covering 1950–2026, filtered down to the 2014–2026 window used in this project.

```
F1 Dataset/
├── circuits.csv
├── constructor_results.csv
├── constructor_standings.csv
├── constructors.csv
├── driver_standings.csv
├── drivers.csv
├── f1_master_2014_2026.csv       # cleaned, merged race-level table
├── f1_features_2014_2026.csv     # + engineered features
├── lap_times.csv
├── pit_stops.csv
├── qualifying.csv
├── races.csv
├── results.csv
├── seasons.csv
├── sprint_results.csv
└── status.csv
```

## Project structure

```
F1 project/
├── F1 Dataset/                        # raw + processed CSVs (see above)
├── filtering.py                       # loads raw CSVs, filters to 2014-2026, merges into one clean table
├── feature_eng.py                     # builds rolling form, teammate-gap, and championship-context features
├── ML_model.py                        # trains + validates the Random Forest classifier, predicts 2026 title odds
├── Result.py                          # generates the final charts and result tables
├── championship_probability.png       # bar chart: win probability by driver
├── championship_predictions_table.png # results table, formatted for sharing
├── points_progression.png             # cumulative points, top 5 contenders, race by race
└── wdc_2026_predictions.csv           # full model output, all drivers
```

## Pipeline

1. **`filtering.py`** — Loads the raw Ergast-schema CSVs, filters races to 2014–2026, and joins results, drivers, constructors, standings, qualifying, and status into one flat table (`f1_master_2014_2026.csv`): one row = one driver, in one race.

2. **`feature_eng.py`** — Engineers the actual model inputs from that table:
   - Rolling driver form (avg finish position/points, DNF rate — last 5 races, always shifted so no future race leaks into the calculation)
   - Rolling constructor (car) strength
   - **Teammate-relative qualifying gap** — the key feature for isolating driver skill from car speed
   - Points gap to the championship leader, and races remaining in the season

3. **`ML_model.py`** — Labels every historical race snapshot with whether that driver went on to win the title that season, then trains a `RandomForestClassifier` on 2014–2025 data. Validated with **season-grouped cross-validation** (never testing on a season the model trained on) using log-loss, to check the predicted probabilities are actually well-calibrated, not just confident-looking. The final model is applied to the current 2026 snapshot to produce win probabilities.

4. **`Result.py`** — Turns the model output into the charts and tables above.

## Key modeling decisions (and why)

- **Grouped by season, not randomly split** — random splitting would let the model "peek" at other races from a season it's meant to be tested on.
- **Every rolling feature is shifted** — a feature can only use races *before* the one being predicted. Skipping this is a common way ML projects quietly cheat during training and fail in the real world.
- **Log-loss over raw accuracy** — the goal is calibrated probabilities ("80% chance" should be right about 80% of the time), not just a single yes/no guess.

## Tech stack

Python · pandas · scikit-learn (RandomForestClassifier, GroupKFold) · matplotlib

## Limitations / next steps

- Standings position and constructor strength currently dominate the model — worth testing how sensitive predictions are to small rank changes (e.g. a tied-on-points driver ranked 2nd vs. 3rd).
- No race-by-race simulation yet — a Monte Carlo layer over the remaining season would turn this into full season-outcome probabilities rather than a single current-standings snapshot.
- Could extend with lap-time/pace data (`lap_times.csv`, `pit_stops.csv`) for a more granular, race-pace-based driver-skill signal.

## Author

Arulalan — [LinkedIn](https://www.linkedin.com/in/arulalan-m/)
