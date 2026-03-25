# Medical Insurance Charges — Linear Regression

A machine learning project that predicts medical insurance costs based on a person's age, BMI, smoking habits, and a few other factors. Built as part of my data science learning journey.


## What this about?

Insurance companies charge different amounts based on a bunch of personal factors. I wanted to see if I could build a model that predicts those charges reasonably well — and understand *which* factors matter most.


## Dataset

- **Source:** [Medical Cost Personal Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Size:** 1,338 rows, 7 columns
- **Features:** age, sex, BMI, number of children, smoker (yes/no), region
- **Target:** insurance charges (USD)

## What I did

1. **Explored the data** — checked distributions, looked for missing values, plotted how each feature relates to charges
2. **Feature engineering** — encoded categorical variables and created a `smoker × bmi` interaction term (smokers with high BMI face the highest charges by far)
3. **Trained a Linear Regression model** using scikit-learn
4. **Evaluated it properly** — not just R², but also MAE, RMSE, and 5-fold cross-validation
5. **Checked residuals** — to see where the model struggles

## Results

The model explains about 86% of the variance in charges — solid for a linear baseline.


## Key finding

Smoking status is the single biggest predictor of charges. When combined with BMI (the `smoker_bmi` interaction), it becomes even more powerful. Non-smokers form a completely separate, much lower charge cluster regardless of other factors.

**Requirements:** pandas, numpy, matplotlib, seaborn, scikit-learn
