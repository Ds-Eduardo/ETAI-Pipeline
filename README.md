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

## Week 7

All rows use ROC-AUC on the development set (5-fold stratified CV); the locked test set is not used. Defaull uses the `config.yaml` parameters; Tuned is nested CV (5 outer folds, 30 Optuna trials per search, 5 inner folds). Logistic regression and the dummy have no search space, so they are only evaluated with their default parameters. No-model floor (dummy, same folds): **0.500**.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std | Chosen hyperparameters |
|---|---|---|---|
| AdaBoost | 0.722 ± 0.014 (+0.008) | 0.728 ± 0.012 | n_estimators=486, learning_rate=0.1515, estimator max_depth=4 |
| Gradient boosting | 0.725 ± 0.015 (+0.028) | 0.727 ± 0.012 | n_estimators=220, learning_rate=0.03432, max_depth=2, min_samples_leaf=6, subsample=0.7877 |
| Oblique random forest | 0.679 ± 0.015 (+0.169) | 0.726 ± 0.012 | tree max_depth=4, tree min_samples_leaf=1, projection n_components=7 |
| HistGradientBoosting | 0.714 ± 0.017 (+0.073) | 0.726 ± 0.014 | max_iter=314, learning_rate=0.01411, max_leaf_nodes=5, min_samples_leaf=85, l2_regularization=2.252 |
| Random forest | 0.682 ± 0.016 (+0.135) | 0.726 ± 0.013 | max_depth=4, min_samples_leaf=8, max_features=0.6226 |
| Logistic regression | 0.723 ± 0.013 (+0.002) | — | — |
| Decision tree | 0.597 ± 0.018 (+0.130) | 0.718 ± 0.011 | max_depth=11, min_samples_leaf=120, criterion=gini |
| Naive Bayes | 0.696 ± 0.012 (+0.002) | 0.703 ± 0.012 | var_smoothing=0.0009348 |
| Dummy (floor) | 0.500 ± 0.000 (+0.000) | — | — |

### Selection metric

I kept ROC-AUC as the selection metric. COMPAS is a risk score used to rank people, and ROC-AUC scores that ranking over every threshold, so it does not depend on the 0.5 cut-off, which is a separate decision. It also has an interpretable floor (0.500 for the dummy, whereas accuracy's floor here is 0.549, the majority-class rate).

## Week 8
### Feature selection (W8)

XGBoost, tuned, ROC-AUC on the development set: nested CV with 5 outer folds, 30 Optuna trials and 5 inner folds. Same outer folds in both runs; the locked test set is not used.

| Run | Nested CV mean ± std | Features kept | Chosen hyperparameters |
|---|---|---|---|
| With selection (VoteSelector, `min_votes` tuned) | 0.726 ± 0.011 | age, priors_count, priors_per_adult_year (3 of 12) | n_estimators=212, learning_rate=0.08265, max_depth=2, min_child_weight=1.177, subsample=0.7439, colsample_bytree=0.6198, reg_lambda=4.976, reg_alpha=0.8586, gamma=3.939, select__min_votes=2 |
| Without selection | 0.728 ± 0.014 | all 12 | n_estimators=292, learning_rate=0.01303, max_depth=6, min_child_weight=15.61, subsample=0.6714, colsample_bytree=0.678, reg_lambda=0.1487, reg_alpha=0.1917, gamma=2.358 |

Corrected paired t-test (with − without): mean difference −0.0012, selection better in 2 of 5 folds, t = −0.39, p = 0.717.

The smaller model is as good as the full one: it gives up 0.001 ROC-AUC, about a tenth of the fold-to-fold std (0.011–0.014), and the corrected test finds no evidence of a difference (p = 0.72). It does so with 3 of 12 features, and its tuned trees are shallower (depth 2 against 6) and more regularised, which fits a signal this weak. I would keep selection on: the score is the same, the model is simpler to explain (age and prior record carry it), and it needs fewer inputs, at the price of a slower fit and one more tuned setting (`min_votes`). Two caveats: with only 5 folds the test has little power, so this is "no detectable loss", not proof of equality; and the kept features come from the final fit on all development rows, so other folds may keep slightly different sets. I would report the nested-CV score, 0.726 ± 0.011 with selection.

| Model | Untuned (roc_auc) | Tuned, nested CV (roc_auc, mean ± std) | Chosen hyperparameters |
|---|---|---|---|
| adaboost | 0.722 | 0.728 ± 0.012 | n_estimators=486, learning_rate=0.1515, estimator__max_depth=4 |
| xgboost | 0.722 | 0.728 ± 0.014 | n_estimators=292, learning_rate=0.01303, max_depth=6, min_child_weight=15.61, subsample=0.6714, colsample_bytree=0.678, reg_lambda=0.1487, reg_alpha=0.1917, gamma=2.358 |
| gradient_boosting | 0.725 | 0.727 ± 0.012 | n_estimators=220, learning_rate=0.03432, max_depth=2, min_samples_leaf=6, subsample=0.7877 |
| xgboost + selection | 0.719 | 0.726 ± 0.011 | n_estimators=212, learning_rate=0.08265, max_depth=2, min_child_weight=1.177, subsample=0.7439,colsample_bytree=0.6198, reg_lambda=4.976, reg_alpha=0.8586, gamma=3.939, select__min_votes=2 |
| oblique_random_forest | 0.679 | 0.726 ± 0.012 | estimator__tree__max_depth=4, estimator__tree__min_samples_leaf=1, estimator__project__n_components=7 |
| hist_gradient_boosting | 0.714 | 0.726 ± 0.014 | max_iter=314, learning_rate=0.01411, max_leaf_nodes=5, min_samples_leaf=85, l2_regularization=2.252 |
| random_forest | 0.682 | 0.726 ± 0.013 | max_depth=4, min_samples_leaf=8, max_features=0.6226 |
| logistic_regression | 0.723 | — | — |
| decision_tree | 0.597 | 0.718 ± 0.011 | max_depth=11, min_samples_leaf=120, criterion=gini |
| naive_bayes | 0.696 | 0.703 ± 0.012 | var_smoothing=0.0009348 |
| dummy | 0.500 | — | — |



## Beyond COMPAS (W9): Adult income

New data, same kind of question: does a person earn more than 50,000 USD a year? The model is trained on `data/adult_train.csv` (6,000 people) and predicts the 2,000 people of `data/adult_new.csv`.

**How to run**

```
python main.py config_adult.yaml    # Adult
python main.py                      # COMPAS, unchanged
```

Adult has its own config and output folder, `results_adult/`: one report per run, the scoreboard `runs.csv`, and `predictions.csv` (`id`, `probability`, and a 0/1 `prediction` at threshold 0.5 for the 2,000 new people; rows out = rows in).

**Choices:** `sex` is the audited attribute; `race` is kept out of the features as well. `fnlwgt` (a sampling weight) and `id` are not features, and `education` is dropped because it duplicates `education_num`. The 25 anonymised `v` columns are all kept as features and left to feature selection.

**What broke, why, and how it was fixed**

| What | Why | Fix |
|---|---|---|
| Engineered features ran on every dataset | their columns are COMPAS's; on Adult they silently did nothing | `preprocessing.engineer_features` flag in the config |
| `main.py` always loaded `config.yaml` | one problem hard-wired | the config path is an argument |
| Default extra column `score_text` | it is COMPAS's own score; Adult has none | `data.extra_columns: []` |
| COMPAS validity rule `age >= 18` | it would erase the 74 people aged 17 | Adult's own rules in the config (`age` 17–90, `hours_per_week` 1–99, gains and losses ≥ 0) |
| `?` read as a category | it is a missing value written as text | `placeholder_tokens` in the config, plus a missingness flag for `workclass` (the same 379 people are missing `occupation`) |
| Fairness text spoke of reoffending | wording tied to recidivism | generic wording: true label 0 but predicted 1 |
| No way to predict unlabeled rows | the pipeline only evaluated | `predict_new_data()`: same cleaning, no de-duplication, same feature split |
| Scoreboard shared between problems | one output folder | own `output` folder per config |

**Results** (XGBoost + feature selection, tuned, nested CV on the development set):

| Metric | Target | No model | Default tree | Ours |
|---|---|---|---|---|
| ROC-AUC | ≥ 0.90 | 0.50 | 0.72 | **0.910 ± 0.015** |
| Accuracy | ≥ 0.855 | 0.76 | 0.80 | **0.861 ± 0.009** |
| F1 (class 1) | ≥ 0.66 | 0.00 | 0.58 | **0.678 ± 0.028** |

Chosen hyperparameters: n_estimators=354, learning_rate=0.05299, max_depth=5, min_child_weight=1.66, subsample=0.74, colsample_bytree=0.5877, reg_lambda=0.3777, reg_alpha=0.02789, gamma=3.764, select__min_votes=1. Features kept (8 of 36): age, education_num, capital_gain, capital_loss, hours_per_week, marital_status, occupation, relationship.

The hardest assumptions to find were: the COMPAS-only engineered features, which did nothing on Adult instead of failing, and the default `score_text` extra column; a hard error would have pointed straight at them. What got us to the target scores was the cleaning and the model more than the tuning: turning `?` into real missing values (with a flag for the people whose workclass and occupation are unknown), target-encoding `native_country`, and a boosted-tree model. With default parameters XGBoost already reached an untuned ROC-AUC of 0.913 ± 0.016, and the tuned nested-CV score (0.910 ± 0.015) is no better, within the fold std. Feature selection kept 8 of 36 columns and none of the 25 anonymised `v` columns, which suggests they add little here. The accuracy margin over the target (0.861 against 0.855) is smaller than its fold std (0.009), so it is a pass but not a comfortable one.

## Week 10
### Missing values

XGBoost, default hyperparameters, feature selection on, ROC-AUC on the development set (5-fold stratified CV, same folds for every imputer). 3.41% of the numeric cells are missing (1,379), mostly in `priors_count` (397) and in the features derived from it; `age` has 120 gaps.

| Imputer | ROC-AUC (mean ± std) | Difference vs median | Folds better than median | p (corrected test) | Time |
|---|---|---|---|---|---|
| median (baseline) | 0.719 ± 0.013 | — | — | — | 10 s |
| KNN | 0.719 ± 0.012 | −0.0003 | 3/5 | 0.782 | 18 s |
| iterative (MICE, one completed dataset) | 0.721 ± 0.011 | +0.0013 | 3/5 | 0.521 | 17 s |
| random forest (missForest) | 0.717 ± 0.011 | −0.0024 | 1/5 | 0.213 | 119 s |

I keep the median imputer. No model-based imputer beats it: the differences (−0.0024 to +0.0013) are at most about a quarter of the fold-to-fold std (0.011–0.013), none wins in more than 3 of 5 folds, and no corrected p-value is below 0.2. The gaps are few and the final model keeps only three features (`age`, `priors_count`, `priors_per_adult_year`), so a smarter guess has little to change, while iterative and KNN cost about 1.7× the time of the median and the random forest about 12× (and warned it had not converged in 10 iterations). With only 5 folds the test has little power, so this means "no detectable gain", not "no difference".

### Predicting new data

```
python main.py                                                   # train, evaluate, save models/model.joblib
python predict.py data/compas_new_people.csv predictions.csv     # predict, no training
```

`predictions.csv` has one row per input row (20 in, 20 out): the `id`, the `probability` of two-year recidivism and the `prediction` (1 if the probability is at least 0.5). 
Result for the 20 new people: 6 are predicted to reoffend (ids 900004, 900006, 900008, 900012, 900014 and 900017) and 14 are not. The mean probability is 0.456, from 0.208 (id 900011) to 0.855 (id 900012). Only two people are clearly above the threshold; 900004 (0.532) and 900008 (0.516) are barely over it, and 900013 (0.474) and the repeated person (0.456) are just under it, so those predictions would flip with a slightly different cut-off. The probabilities are a ranking, not a verdict, and an overall AUC of about 0.72 means many individual calls are uncertain.

