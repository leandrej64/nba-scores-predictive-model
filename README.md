# NBA Score Predictive Modelling

Predicting NBA team point distributions from historical game statistics.

## Data

~15,000 NBA matches from 2020, with team/game-level statistics. Provided as part of a take-home modelling challenge for [Sporting Risk](https://www.sportingrisk.com/) (London), interview round 3.

## Methods

Two models, built entirely from scratch (no scikit-learn), each producing a full predicted probability distribution of team points rather than a single estimate:

- **LASSO regression** (gradient descent, L1-penalized) on rolling team/opponent statistics, with a Normal predictive distribution.
- **Random Forest** (custom bagging + feature bagging), trained on LASSO's selected features, with an empirical per-leaf distribution.

Both are cross-validated with time-ordered folds (no data leakage) and scored with the Ranked Probability Score (RPS).

## Results

Skill score (RPSS) vs. a naive baseline (Normal distribution centered on a team's last-10-game average), on the test set:

| Model | RPSS |
|---|---|
| LASSO | 6.61% |
| Random Forest | 6.2% |

Full methodology, results, and discussion in [`report.pdf`](report.pdf).
