# ScratchML

ScratchML is a compact machine learning library built from first principles with NumPy. The project focuses on implementing classic algorithms by hand while keeping the code readable, testable, and close to familiar scikit-learn style APIs.

## Why this project stands out

- Consistent estimator interface across regression, classification, clustering, and ensemble models.
- Core algorithms implemented manually instead of wrapping existing ML libraries.
- Unit tests for correctness and API behavior.
- A runnable demo script that benchmarks the scratch implementations on synthetic datasets.
- Clean package structure that is easy to extend with new models.

## Implemented models

- Linear models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNetRegression`
- Classification: `LogisticRegression`, `SoftmaxRegression`
- Nearest neighbors: `KNeighborsClassifier`, `KNeighborsRegressor`
- Preprocessing: `StandardScaler`, `MinMaxScaler`
- Trees and ensembles: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`
- Clustering: `KMeans`

## Project structure

```text
scratchml/
  __init__.py
  cluster.py
  ensemble.py
  linear_model.py
  neighbors.py
  preprocessing.py
  tree.py
  utils.py
tests/
  test_linear_models.py
  test_preprocessing.py
  test_tree_ensemble_and_cluster.py
notebooks/
  examples_and_sklearn_comparisons.ipynb
main.py
pyproject.toml
README.md
```

## Quick start

```bash
python3 -m unittest discover -s tests -v
python3 main.py
```

## Notebook walkthrough

An example notebook with end-to-end use cases and optional `scikit-learn` comparisons lives at [notebooks/examples_and_sklearn_comparisons.ipynb](/Users/prateekbhandari/Desktop/projects/ml_from_scratch/notebooks/examples_and_sklearn_comparisons.ipynb).

## Design goals

- Keep the math visible.
- Favor readable code over premature optimization.
- Follow conventions that make the library feel professional.
- Support simple educational experiments without hidden magic.

## Next ideas

- Add feature scaling and preprocessing helpers.
- Add model serialization and plotting utilities.
- Add notebook examples explaining the derivations behind each algorithm.
- Add cross-validation and hyperparameter search utilities.
