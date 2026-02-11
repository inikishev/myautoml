<!-- # ![tests](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg) -->

<h1 align='center'>myautoml</h1>

Library that I use for kaggle competitions. This is kind of like Autogluon but you decide what models you fit, and all models you fit are saved for later usage in ensembling, stacking, etc.

Following Autogluon's strategy (since it seeme to win all AutoML benchmarks), I use synchronized stratified k-folds and all ensembling is done on out-of-fold predictions, and for inference per-fold predictions are averaged. Actually in autogluon ensembling is fitted on averaged predictions, whereas I only average at the very end which I think should work better. But I literally just made this and now I am going to benchmark it and see whether it is better.

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

#### 2. Fit L1 models

Fit any model with sklearn-compatible fit and predict methods.

```python
fitter.fit_model(
    name = "LR L1",
    model = make_pipeline(StandardScaler(), LogisticRegression())
)
fitter.fit_model(
    name = "DT L1",
    model = make_pipeline(StandardScaler(), DecisionTreeClassifier())
)
fitter.fit_model(
    name = "RF L1",
    model = make_pipeline(StandardScaler(), RandomForestClassifier()),

    # merge some folds if model is slow to fit.
    # since folds are just merged, they are still synchronized
    # with the unmerged folds.
    max_folds=4,
)
# ... etc
```

#### 3. Fit ensembles / stacks

To reduce leakage as much as possible, all ensemble models are fitted on out-of-fold predictions of underlying models. For inference only the per-fold predictions of the very last model are averaged.

Ensemble can be fitted to just predictions of models of previous stack level, or predictions of all models of all stack levels, or predictions plus original features.

```python
l1_models = ["LR L1", "DT L1"]
# or l1_models = fitter.select_models(stack_level=1)

# fit a weighted ensemble, e.g. using RidgeClassifierCV on predictions
fitter.fit_model(
    name = "WeightedEnsemble L2",
    model = make_pipeline(StandardScaler(), RidgeClassifierCV()),
    stack_models = l1_models,
    passthrough = False # only fit on predictions of previous models
)

# fit stack models with both original features and predictions of L1 models
fitter.fit_model(
    name = "LR L2",
    model = make_pipeline(StandardScaler(), LogisticRegression()),
    stack_models = l1_models,
)
fitter.fit_model(
    name = "DT L2",
    model = make_pipeline(StandardScaler(), DecisionTreeClassifier()),
    stack_models = l1_models,
)
# ... etc

# Fit L3 models
l2_models = ["LR L2", "DT L2", "WeightedEnsemble L2"]
fitter.fit_model(
    name = "WeightedEnsemble L3",
    model = make_pipeline(StandardScaler(), RidgeClassifierCV()),
    stack_models = l2_models,
    passthrough = False
)
# ...

```

#### 4. Fitting transformers

myautoml also supports fitting feature transformers, such as feature selection, PCA, kernel approximations, etc. This is very useful for transformers that are expensive to fit, such as ``SequentialFeatureSelector`` or iterative imputers.

A transformer can be fitted to all data, or to each fold to prevent leakage for transforms that use labels (like LDA and PLSS). Then, when you fit a model, it is fitted to out-of-fold predictions of the transformer.

It is also possible fit a transformer to predictions of other models with or without passthrough, and to outputs of another transformer.

```python
# fit SequentialFeatureSelector
fitter.fit_transformer(
    name = "SFF",
    transformer = make_pipeline(
        StandardScaler(),
        SequentialFeatureSelector(LogisticRegression()),
        scoring="roc_auc",
    ),
    use_folds = False,
)

# now we can fit models with this transformer
fitter.fit_model(
    name = "SFF-LR L1",
    model = make_pipeline(StandardScaler(), LogisticRegression()),
    transformer = "SFF",
)

# transformer can also be fitted to predictions
# of other models rather than dataset features, or to both
fitter.fit_transformer(
    name = "SFF L2",
    transformer = make_pipeline(
        StandardScaler(),
        SequentialFeatureSelector(LogisticRegression()),
        scoring="roc_auc",
    )
    use_folds = False,
    stack_models = l2_models,
    passthrough = False,
)

fitter.fit_model(
    name = "SFF-LR L3",
    model = make_pipeline(StandardScaler(), LogisticRegression()),
    transformer = "SFF L2",
)
```

#### 5. Inference

```python
probas = fitter.predict_proba(df_test, model="SFF-LR L3")
# numpy array of shape (n_samples, n_classes), only for classification

preds = fitter.predict(df_test, model="SFF-LR L3")
# Returns a polars Series.
```

#### Other stuff

WIP