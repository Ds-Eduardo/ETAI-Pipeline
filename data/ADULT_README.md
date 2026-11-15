# Adult income data (for the "new dataset" challenge)

**Task:** predict whether a person earns more than 50,000 USD a year (`income_over_50k`: 1 = yes, 0 = no) from census information.

| File | Rows | Target |
|---|---|---|
| `adult_train.csv` | 6,000 people | yes (`income_over_50k`) |
| `adult_new.csv` | 2,000 new people | no: these are the rows your final model predicts |

Each row has an `id`. About 24% of people in `adult_train.csv` earn over 50K.

## Columns

| Column | Description |
|---|---|
| `id` | row identifier (not a feature) |
| `age` | age in years |
| `workclass` | type of employer (private, self-employed, government, ...) |
| `fnlwgt` | census sampling weight: how many people in the population this row represents |
| `education` | highest education level, as text |
| `education_num` | highest education level, as a number |
| `marital_status` | marital status |
| `occupation` | type of occupation |
| `relationship` | role in the household (husband, wife, own child, not in family, ...) |
| `race` | race |
| `sex` | sex |
| `capital_gain` | capital gains in the year |
| `capital_loss` | capital losses in the year |
| `hours_per_week` | usual working hours per week |
| `native_country` | country of origin |
| `v01` ... `v25` | additional anonymised attributes supplied with the data; their meaning is not documented |
| `income_over_50k` | **target** (train file only) |

Look at the data before configuring the pipeline: some columns have placeholder values, missing values or text that needs cleaning, and some columns may not be useful at all.

**Source:** derived from the UCI Adult dataset (Becker & Kohavi, 1996; 1994 US Census), licensed CC BY 4.0. Subsampled and extended for this course.
