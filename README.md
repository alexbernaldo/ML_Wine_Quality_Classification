# ML_Wine_Quality_Classification

Classification of Spanish wines from physicochemical features — predicting **wine quality**
(multi-class, 3–9) and **wine type** (binary red/white).

## Dataset

`wines.csv` (UCI-style Spanish wine dataset) — 6,497 samples. Features: `quality`,
`wine_type`, `alcohol`, `fixed_acidity`, `volatile_acidity`, `pH`, `sulphates`, `density`,
`residual_sugar`, `chlorides`, sulfur dioxides, `region_class` (DO/DOCa), `location`, `price`.

## Approach

Two exercises sharing five classifiers — Decision Tree, k-NN, Support Vector Machine,
Gaussian Naive Bayes, Random Forest:

- **Exercise 1 — wine quality (multi-class):** hyperparameter tuning (8+ configs each),
  correlation-based feature selection (|r| ≥ 0.26), PCA, minimal attribute subset study.
- **Exercise 2 — red vs. white (binary, stratified 85/15):** same classifiers plus
  detection of a possible intermediate category via Euclidean distance to centroids and PCA.

## Results

| Exercise | Best model | Accuracy | Macro F1 |
|---|---|---|---|
| E1 — quality | Decision Tree / SVM / RF | ~1.00 (0.995–1.00) | 0.95–0.96 |
| E1 — baseline kNN | k=5, Manhattan | 0.90 | 0.68 |
| E2 — type | DT / SVM / GNB / RF | 1.00 | 1.00 |

Key finding: a small correlation-selected feature subset matches full-model accuracy;
no stable third wine class exists (650 of 6,497 samples fall in a "middle" band).

## Reproduce

Download wines.csv, open the relevant exercise notebook (PW1_E1_GROUP1.ipynb or
PW1_E2_GROUP1.ipynb) in Jupyter/Colab, and run cells top to bottom.
