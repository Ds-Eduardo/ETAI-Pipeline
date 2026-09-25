## 20260493, Eduardo Cavalcante

# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Data cleaning & feature engineering | Added data validation and cleaning, duplicate removal, and deterministic engineered predictors; compared baseline and engineered feature sets with nested cross-validation while keeping the locked test set out of model selection. |
| 4 | Hyperparameter tuning & ensemble models | Added decision-tree, random-forest, oblique-random-forest, and AdaBoost comparisons; evaluates default CV and tuned nested CV on the development set, reports fold-to-fold variation and train-validation gaps, and records the best trial and parameters. |


- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.


## Comparison between LR and DT
Decision tree seemms to be suffering from overfitting, having more 20% of accuracy in the train set than test set. While logistic regression shows more stable and better performance on the test set (+5%).


## WEEK 3
After cleaning the dataset and running the decision tree, the model’s predictive power decreased, although this was practically negligible (around 2%, from 63.4% to 61.3% on the test set). Furthermore, the decision tree continued to show signs of overfitting.
The same was true for logistic regression. There was a negligible decrease in predictive power (67.8% to 66.5%); however, it remains the better option of the two models used.



## Feature engineering comparison

`add_engineered_features()` creates `juvenile_total` (the sum of the three juvenile offense counts), `priors_per_year` (`priors_count / age`), and `age_x_log_priors` (`age * log1p(priors_count)`). These are deterministic, row-level transformations and learn nothing from the data, so they can be computed before the development/test split. Missing inputs remain missing for imputation inside the pipeline.

The nested comparison uses the same five outer folds for each feature set and model, with three inner folds to tune each model. It is run on the development set; the locked test set is not scored.

| Model | Baseline accuracy (mean ± fold SD) | Engineered accuracy (mean ± fold SD) | Mean gain |
|-------|------------------------------------|--------------------------------------|-----------|
| Logistic regression | 0.660 ± 0.014 | 0.670 ± 0.016 | +0.009 |
| Random forest | 0.668 ± 0.017 | 0.664 ± 0.021 | -0.004 |
| AdaBoost | 0.672 ± 0.021 | 0.666 ± 0.019 | -0.006 |

Logistic regression is the only family with a positive mean gain, but its gain is smaller than its baseline fold SD. No family gains by more than the fold SD.

### Ensembles

All rows use accuracy on the development set: default five-fold CV and, where a search space is configured, five-fold nested CV with 30 Optuna trials per inner search. The locked test set is not used.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.675 ± 0.011 (+0.002) | — | — | — |
| Decision tree | 0.600 ± 0.020 (+0.103) | 0.672 ± 0.005 (+0.011) | 0.673 | max_depth=4, min_samples_leaf=1 |
| Random forest | 0.648 ± 0.017 (+0.101) | 0.671 ± 0.010 (+0.017) | 0.679 | max_depth=11, min_samples_leaf=8, max_features=0.2372 |
| Oblique random forest | 0.642 ± 0.013 (+0.135) | 0.676 ± 0.013 (+0.012) | 0.679 | tree max_depth=6, tree min_samples_leaf=15, projection n_components=8 |
| AdaBoost | 0.678 ± 0.008 (+0.003) | 0.676 ± 0.007 (+0.009) | 0.680 | n_estimators=230, learning_rate=0.1568, estimator max_depth=4 |

I would choose tuned AdaBoost as a practical candidate: its nested-CV mean is 0.676, tied with oblique random forest at the displayed precision, and its 0.005 lead over random forest is smaller than its fold-to-fold SD of 0.007. The depths are broadly plausible: the standalone tree is capped at depth 4, while the ordinary forest can use deeper trees (depth 11) because averaging reduces variance, and its minimum leaf size of 8 regularizes individual trees. The oblique forest uses depth 6 and leaves of at least 15 after projecting features, while AdaBoost's depth-4 learners are more complex than classic stumps but use a low learning rate of 0.1568 over 230 estimators. I would report AdaBoost's nested-CV accuracy of **0.676 ± 0.007** (gap +0.009) for the tuned procedure, not the best-trial score of 0.680, which is optimistically selected.
