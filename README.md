# Optimizing Wastewater Treatment Through AI

A machine learning project that uses water-quality parameters to predict whether a water sample is **potable (drinkable)** or **non-potable**. The notebook performs exploratory data analysis, data preprocessing, model training, evaluation, and hyperparameter tuning using several classification algorithms.

## Project Overview

Water quality can be evaluated using measurable parameters such as pH, turbidity, dissolved oxygen, temperature, and total dissolved solids (TDS). This project applies machine learning to these parameters to classify water samples according to their `Potability` label.

The main workflow is:

1. Load the custom water-quality dataset.
2. Explore and inspect the dataset.
3. Handle missing values using column means.
4. Visualize distributions, relationships, and correlations.
5. Split the data into training and testing sets.
6. Train multiple classification models.
7. Evaluate models using accuracy, confusion matrices, precision, recall, and F1-score.
8. Perform hyperparameter tuning for Decision Tree and KNN models.
9. Compare model performance.

## Dataset

The notebook loads:

```text
custom-dataset.csv
```

The dataset contains **5,000 records** and the following columns:

| Feature            | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `pH`               | Acidity/alkalinity level of the water                |
| `Turbidity`        | Measure related to water clarity                     |
| `Dissolved Oxygen` | Amount of dissolved oxygen in the water              |
| `Temperature`      | Water temperature                                    |
| `TDS`              | Total dissolved solids                               |
| `Potability`       | Target label indicating whether the water is potable |

The target column is `Potability`, while the other five columns are used as model features.

## Data Preprocessing

The notebook performs the following preprocessing steps:

* Checks the dataset shape and data types.
* Checks for missing values.
* Replaces missing values with the corresponding column mean.
* Examines unique values and descriptive statistics.
* Separates features (`X`) from the target (`Y`).
* Splits the data into:

  * **67% training data:** 3,350 samples
  * **33% test data:** 1,650 samples
* Uses `random_state=42` for the train/test split.

Feature normalization with `StandardScaler` appears in the notebook as commented-out code and is therefore not part of the executed preprocessing workflow.

## Exploratory Data Analysis

The notebook includes visual analysis such as:

* Potability class distribution
* pH distribution
* Violin plots
* Box plots for feature inspection and outliers
* Histograms
* Pair plots
* pH vs. potability scatter plot
* Correlation heatmap
* Dataset box plots

These visualizations are used to understand the structure of the dataset and relationships between water-quality parameters and potability.

## Machine Learning Models

The project evaluates the following classifiers:

* Decision Tree
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Random Forest
* XGBoost
* Support Vector Machine (SVM)

### Test Accuracy

| Model               | Test Accuracy |
| ------------------- | ------------: |
| Decision Tree       |        94.00% |
| KNN                 |        93.21% |
| Logistic Regression |        95.76% |
| Random Forest       |        92.91% |
| XGBoost             |    **96.73%** |
| SVM                 |        93.88% |

Based on the reported test results in the notebook, **XGBoost achieved the highest reported accuracy of 96.73%**.

## Model Details

### Decision Tree

Initial configuration:

```python
DecisionTreeClassifier(
    criterion="entropy",
    min_samples_split=9,
    splitter="best"
)
```

Reported initial test accuracy:

```text
94.00%
```

The notebook subsequently performs GridSearchCV-based tuning using:

* `criterion`: `gini`, `entropy`
* `splitter`: `best`, `random`
* `min_samples_split`: tested values

The tuned Decision Tree used later in the notebook reports:

```text
94.55% test accuracy
```

### KNN

Initial configuration:

```python
KNeighborsClassifier(
    metric="euclidean",
    n_neighbors=24,
    weights="uniform"
)
```

Reported initial test accuracy:

```text
93.21%
```

Grid search identifies:

```python
{
    "metric": "euclidean",
    "n_neighbors": 5,
    "weights": "distance"
}
```

The tuned KNN reports approximately:

```text
95.39% test accuracy
```

### Logistic Regression

Configuration:

```python
LogisticRegression(
    max_iter=120,
    random_state=0,
    n_jobs=20
)
```

Reported test accuracy:

```text
95.76%
```

### Random Forest

Configuration:

```python
RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=0.16,
    random_state=42
)
```

Reported test accuracy:

```text
92.91%
```

### XGBoost

Configuration:

```python
XGBClassifier(
    max_depth=8,
    n_estimators=125,
    random_state=0,
    learning_rate=0.03,
    n_jobs=5
)
```

Reported test accuracy:

```text
96.73%
```

The classification report reports:

* `False` class F1-score: **0.72**
* `True` class F1-score: **0.98**
* Weighted F1-score: **0.96**

### Support Vector Machine

Configuration:

```python
SVC(
    kernel="rbf",
    random_state=42
)
```

Reported test accuracy:

```text
93.88%
```

## Decision Tree Feature Importance

The initial Decision Tree reports the following feature importances:

| Feature          | Importance |
| ---------------- | ---------: |
| Turbidity        |     0.3334 |
| pH               |     0.3243 |
| Dissolved Oxygen |     0.2548 |
| TDS              |     0.0511 |
| Temperature      |     0.0364 |

These values describe the importance assigned by that particular Decision Tree model; they should not be interpreted as causal effects.

## Example Prediction

The notebook demonstrates prediction for a single sample:

```python
dt.predict([[7.59, 4.29, 7.98, 25, 42]])
```

The executed notebook returns:

```text
[True]
```

The five values correspond to:

```text
pH
Turbidity
Dissolved Oxygen
Temperature
TDS
```

## Evaluation

The notebook evaluates the models using:

* Accuracy
* Confusion matrix
* Precision
* Recall
* F1-score
* Classification report

For the XGBoost model, the reported classification metrics are:

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| False |      0.93 |   0.58 |     0.72 |
| True  |      0.97 |   1.00 |     0.98 |

Overall:

* Accuracy: **0.97**
* Macro F1-score: **0.85**
* Weighted F1-score: **0.96**

## Hyperparameter Tuning

The notebook uses `GridSearchCV` and `RepeatedStratifiedKFold` for model optimization.

### Decision Tree

The grid search evaluates different combinations of:

```text
criterion
splitter
min_samples_split
```

The search reports:

```python
{
    "criterion": "entropy",
    "min_samples_split": 9,
    "splitter": "random"
}
```

A subsequently configured Decision Tree reports **94.55%** test accuracy.

### KNN

The KNN grid search evaluates:

```text
n_neighbors: 1–30
weights: uniform, distance
metric: euclidean, manhattan, minkowski
```

Best cross-validation parameters reported by the notebook:

```python
{
    "metric": "euclidean",
    "n_neighbors": 5,
    "weights": "distance"
}
```

The resulting test accuracy is approximately **95.39%**.

## Technologies Used

* Python
* Jupyter Notebook / Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* XGBoost

## Installation

Clone the repository and install the required libraries:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
```

Make sure `custom-dataset.csv` is available in the location expected by the notebook.

## Running the Project

### Using Jupyter Notebook

```bash
jupyter notebook Optimise_Waste_Water_Treatment_Through_AI.ipynb
```

### Using Google Colab

Upload:

```text
Optimise_Waste_Water_Treatment_Through_AI.ipynb
custom-dataset.csv
```

Then run the notebook cells sequentially.

## Project Structure

```text
.
├── Optimise_Waste_Water_Treatment_Through_AI.ipynb
├── custom-dataset.csv
└── README.md
```

## Results Summary

The notebook demonstrates that several machine learning approaches can classify water samples using the available water-quality parameters.

The highest test accuracy explicitly reported in the executed model evaluations is:

```text
XGBoost — 96.73%
```

The results should be interpreted in the context of this particular custom dataset and the notebook's train/test split. Accuracy alone does not establish real-world drinking-water safety, and deployment would require appropriate validation against domain standards and representative operational data.

## Future Improvements

Possible extensions include:

* Add a dedicated validation dataset.
* Address the class imbalance between potable and non-potable samples.
* Compare additional metrics such as ROC-AUC, PR-AUC, and balanced accuracy.
* Perform more systematic hyperparameter optimization for all models.
* Add feature scaling where appropriate for distance- and kernel-based models.
* Save trained models for later inference.
* Build a simple web application for water-quality prediction.
* Add automated data validation and preprocessing pipelines.
* Evaluate the model on independent real-world water-quality data.

## License

No license is specified in the notebook. Add an appropriate license file before distributing the project publicly.



