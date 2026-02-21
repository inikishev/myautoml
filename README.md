<!-- # ![tests](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg) -->

<h1 align='center'>myautoml</h1>

Library that I use for kaggle competitions. This is kind of like Autogluon but you decide what models you fit, and all models you fit are saved for later usage in ensembling, stacking, etc.

Following Autogluon's strategy (since it seeme to win all AutoML benchmarks), I use synchronized stratified k-folds and all ensembling is done on out-of-fold predictions, and for inference per-fold predictions are averaged. Actually in autogluon ensembling is fitted on averaged predictions, whereas I only average at the very end which I think should work better.

> **note:** While I am developing this, updates may introduce API changes that can break existing folders.

### How to use


#### 1. Initialize

First time ``fitter.initialize`` is ran, it creates a directory where all fitted models are saved as well as other stuff such as fold indexes. The next time the directory will be loaded and all models will be there, and you can continue fitting new models. No model you fit goes to waste - ensembles from many diverse models are extremely powerful.

```python
import polars as pl
import myautoml as ma

# load some data
df_train = pl.read_csv("train.csv")
df_test = pl.read_csv("test.csv")

# create or load a fitter
fitter = ma.TabularFitter()
fitter.initialize(df_train, y="Heart Disease", X_unlabeled=df_test, eval_metric='roc_auc', n_folds=8, drop_cols='id')
```

#### 2. Fit base models

Fit any model with sklearn-compatible fit and predict methods.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

errors = fitter.fit_supervised(
    name="RF",
    estimator=RandomForestClassifier(),
)

errors = fitter.fit_supervised(
    name="GB",
    estimator=GradientBoostingRegressor(),
)

fitter.list_fitted()

preds = fitter.predict(X_new, estimator="RF")
proba = fitter.predict_proba(X_new, estimator="RF")
```

#### 3. Stacking Models

Models can be fitted to out-of-fold predictions of other models (stacking)

```python
fitter.fit_supervised(
    name="RF L1",
    estimator=RandomForestClassifier(),
    inputs=["RF", "GB"],  # Fit to out-of-fold predictions of RF and FB
)

fitter.fit_supervised(
    name="RF L1-passthrough",
    estimator=RandomForestClassifier(),
    # None = original features
    inputs=[None, "RF", "GB"],  # Fit to original features and out-of-fold predictions
)


# Use helper to select estimators for stacking
fitter.fit_supervised(
    name="GB L1",
    estimator=GradientBoostingRegressor(),
    inputs=fitter.select_estimators(stack_level=0) # selects ["RF", "GB"]
)
```

#### Unsupervised Estimators / Feature Transformers

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Fit unsupervised transformer
fitter.fit_unsupervised(
    name="KMeans",
    estimator=KMeans(5),
    use_folds=True,        # Fit to each fold to avoid data leakage
    method="predict", # method to call on KMeans
)

# Fit unsupervised transformer to out-of-fold predictions
fitter.fit_unsupervised(
    name="PCA L1",
    estimator=PCA(n_components=10),
    use_folds=True,
    method="transform",
    inputs=["RF", "GB"],
)

# Use transformed features in supervised model
fitter.fit_supervised(
    name="PCA-L1 RF",
    estimator=RandomForestClassifier(),
    inputs=[None, "PCA L1", "KMeans"],  # Original features + PCA + KMeans
)
```

#### Other useful methods

```python
# View a summary of fitted estimators with their score
df = fitter.list_fitted(sort="score_test_mean")

# Rename an estimator
fitter.rename("RF", "RandomForest")
```
