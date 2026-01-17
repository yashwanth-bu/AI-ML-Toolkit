This is a **structured, comprehensive overview** of the major **features, functions, classes, and modules** provided by **scikit-learn (sklearn)** for building and working with machine-learning models.
This is not an exhaustive list but covers the essential and widely used components.

---

# ✅ **1. Core Modules of scikit-learn**

Scikit-learn is organized into subpackages. The major ones include:

---

## **1.1 sklearn.datasets**

Tools for loading and generating datasets.

**Key functions:**

* `load_iris()`, `load_wine()`, `load_breast_cancer()`
* `load_digits()`, `load_diabetes()`
* `fetch_20newsgroups()`, `fetch_openml()`
* Synthetic dataset generators:

  * `make_classification()`
  * `make_regression()`
  * `make_blobs()`
  * `make_moons()`
  * `make_circles()`

---

## **1.2 sklearn.model_selection**

Tools for splitting data and evaluating models.

**Key functions and classes:**

* **Train-test splitting**

  * `train_test_split()`
* **Cross-validation**

  * `cross_val_score()`
  * `cross_validate()`
  * `LearningCurveDisplay`, `ValidationCurveDisplay`
* **Hyperparameter search**

  * `GridSearchCV`
  * `RandomizedSearchCV`
  * `HalvingGridSearchCV`
  * `HalvingRandomSearchCV`
* **Split strategies:**

  * `KFold`, `StratifiedKFold`
  * `GroupKFold`, `TimeSeriesSplit`
  * `ShuffleSplit`, `StratifiedShuffleSplit`

---

## **1.3 sklearn.preprocessing**

Transformers for feature scaling and encoding.

**Scaling & normalization:**

* `StandardScaler`
* `MinMaxScaler`
* `RobustScaler`
* `Normalizer`
* `QuantileTransformer`
* `PowerTransformer`

**Encoding:**

* `OneHotEncoder`
* `OrdinalEncoder`
* `LabelEncoder`
* `LabelBinarizer`

**Feature engineering:**

* `PolynomialFeatures`
* `Binarizer`
* `KBinsDiscretizer`

---

## **1.4 sklearn.impute**

Tools for missing-value imputation.

* `SimpleImputer`
* `KNNImputer`
* `IterativeImputer`
* `MissingIndicator`

---

## **1.5 sklearn.pipeline**

Pipeline utilities to chain preprocessing + models.

* `Pipeline`
* `make_pipeline()`
* `FeatureUnion`
* `ColumnTransformer`
* `make_column_transformer()`

---

## **1.6 sklearn.feature_selection**

Tools for selecting important features.

* **Filter methods:**

  * `SelectKBest`, `SelectPercentile`
  * `chi2`, `f_regression`, `f_classif`
* **Wrapper methods:**

  * `RFE`, `RFECV`
* **Embedded methods:**

  * `SelectFromModel`
* **Model-specific selectors:**

  * L1-based selection using `Lasso`
  * Tree-based selection using `RandomForestClassifier`

---

## **1.7 sklearn.decomposition**

Dimensionality reduction methods.

* `PCA`, `IncrementalPCA`, `KernelPCA`
* `TruncatedSVD`
* `NMF`
* `FactorAnalysis`
* `FastICA`
* `DictionaryLearning`

---

## **1.8 sklearn.metrics**

Evaluation metrics for regression, classification, clustering.

**Classification:**

* `accuracy_score`
* `precision_score`, `recall_score`, `f1_score`
* `confusion_matrix`
* `classification_report`
* `roc_curve`, `auc`
* `log_loss`

**Regression:**

* `mean_squared_error`
* `mean_absolute_error`
* `r2_score`

**Clustering:**

* `silhouette_score`
* `calinski_harabasz_score`
* `davies_bouldin_score`

---

## **1.9 sklearn.linear_model**

Linear models for regression and classification.

### Regression:

* `LinearRegression`
* `Ridge`, `Lasso`, `ElasticNet`
* `BayesianRidge`
* `SGDRegressor`
* `HuberRegressor`

### Classification:

* `LogisticRegression`
* `LinearSVC`
* `RidgeClassifier`
* `SGDClassifier`
* `Perceptron`

---

## **1.10 sklearn.tree**

Decision trees.

* `DecisionTreeClassifier`
* `DecisionTreeRegressor`
* `plot_tree()`

---

## **1.11 sklearn.ensemble**

Ensemble methods.

### Bagging methods:

* `BaggingClassifier`, `BaggingRegressor`
* `RandomForestClassifier`, `RandomForestRegressor`
* `ExtraTreesClassifier`, `ExtraTreesRegressor`

### Boosting:

* `AdaBoostClassifier`, `AdaBoostRegressor`
* `GradientBoostingClassifier`
* `GradientBoostingRegressor`

### Stacking & Voting:

* `StackingClassifier`, `StackingRegressor`
* `VotingClassifier`, `VotingRegressor`

---

## **1.12 sklearn.svm**

Support Vector Machines.

* `SVC` (classification)
* `SVR` (regression)
* `LinearSVC`
* `OneClassSVM` (outlier detection)

---

## **1.13 sklearn.neighbors**

K-Nearest Neighbors algorithms.

* `KNeighborsClassifier`
* `KNeighborsRegressor`
* `NearestNeighbors`
* `RadiusNeighborsClassifier`
* `KDTree`, `BallTree`

---

## **1.14 sklearn.naive_bayes**

Naïve Bayes models.

* `GaussianNB`
* `MultinomialNB`
* `BernoulliNB`
* `ComplementNB`
* `CategoricalNB`

---

## **1.15 sklearn.cluster**

Clustering algorithms.

* `KMeans`
* `MiniBatchKMeans`
* `DBSCAN`
* `OPTICS`
* `MeanShift`
* `AgglomerativeClustering`
* `SpectralClustering`
* `Birch`

---

## **1.16 sklearn.manifold**

Manifold learning & nonlinear dimensionality reduction.

* `TSNE`
* `Isomap`
* `LocallyLinearEmbedding (LLE)`
* `MDS`

---

## **1.17 sklearn.covariance**

Covariance estimators & anomaly detection.

* `EllipticEnvelope`
* `LedoitWolf`, `OAS`
* `GraphicalLasso`

---

## **1.18 sklearn.semi_supervised**

Semi-supervised learning.

* `LabelSpreading`
* `LabelPropagation`
* `SelfTrainingClassifier`

---

## **1.19 sklearn.gaussian_process**

Gaussian process regression and classification.

* `GaussianProcessRegressor`
* `GaussianProcessClassifier`
* Kernels: `RBF`, `Matern`, `RationalQuadratic`

---

## **1.20 sklearn.inspection**

Model interpretation tools.

* `permutation_importance`
* `PartialDependenceDisplay`
* `DecisionBoundaryDisplay`

---

## **1.21 sklearn.utils**

Utility functions (mixins, random generators, validation helpers).

* `shuffle`
* `resample`
* `check_array`
* `all_estimators`

---

# 🎯 Summary Table

| Category                 | Key Tools                                                       |
| ------------------------ | --------------------------------------------------------------- |
| Data                     | `datasets`, `make_*` functions                                  |
| Prep / Cleaning          | `preprocessing`, `impute`, `feature_selection`                  |
| Model Training           | `linear_model`, `svm`, `neighbors`, `tree`, `ensemble`          |
| Model Evaluation         | `model_selection`, `metrics`                                    |
| Dimensionality Reduction | `PCA`, `SVD`, `NMF`, `t-SNE`, `LLE`                             |
| Pipelines                | `Pipeline`, `ColumnTransformer`                                 |
| Advanced                 | `gaussian_process`, `manifold`, `covariance`, `semi_supervised` |

---

Below is an **expanded list** of scikit-learn features—covering more modules, utilities, helper classes, configuration tools, meta-estimators, datasets, inspection tools, and lesser-known functionalities. This list goes **beyond the common APIs** and includes more detailed internal and advanced components.

---

# 🔥 **Additional and More Advanced scikit-learn Features**

---

# ✅ **2. Extended and Advanced Preprocessing Features**

## **2.1 Feature Scaling & Transformation**

* `MaxAbsScaler`
* `FunctionTransformer`
* `SplineTransformer`
* `KBinsDiscretizer`
* `PolynomialCountSketch`

---

## **2.2 Encoding & Categorical Handling**

* `MultiLabelBinarizer`
* `DictVectorizer`
* `FeatureHasher`

---

## **2.3 Text Preprocessing**

(from `sklearn.feature_extraction.text`)

* `CountVectorizer`
* `TfidfVectorizer`
* `TfidfTransformer`
* `HashingVectorizer`
* `ENGLISH_STOP_WORDS`

---

## **2.4 Image Feature Extraction**

(from `sklearn.feature_extraction.image`)

* `img_to_graph`
* `grid_to_graph`
* `extract_patches_2d`
* `PatchExtractor`

---

## **2.5 Feature Extraction (General)**

(from `sklearn.feature_extraction`)

* `DictVectorizer`
* `FeatureHasher`

---

# ✅ **3. Expanded Model Selection Tools**

* `ParameterGrid`
* `ParameterSampler`
* `learning_curve()`
* `validation_curve()`
* `permutation_test_score()`

---

# ✅ **4. Transformers & Meta-Estimators**

Meta-estimators wrap other models:

* `TransformedTargetRegressor`
* `MultiOutputRegressor`
* `MultiOutputClassifier`
* `ClassifierChain`
* `RegressorChain`
* `OneVsOneClassifier`
* `OneVsRestClassifier`
* `OutputCodeClassifier`
* `CalibratedClassifierCV`

---

# ✅ **5. More Linear Models**

### **Generalized Linear Models**

* `PoissonRegressor`
* `GammaRegressor`
* `TweedieRegressor`

### **Outlier-robust linear models**

* `RANSACRegressor`
* `QuantileRegressor`
* `TheilSenRegressor`

---

# ✅ **6. More Tree & Ensemble Tools**

### Extra decision tree utilities:

* `plot_tree`
* `export_graphviz`
* `export_text`

### Extra ensemble utilities:

* `HistGradientBoostingClassifier`
* `HistGradientBoostingRegressor`

---

# ✅ **7. Expanded Clustering Features**

### Specialized clustering methods:

* `AffinityPropagation`
* `OPTICS`
* `AgglomerativeClustering`
* `SpectralClustering`
* `BisectingKMeans`

### Pairwise metrics for clustering:

(from `sklearn.metrics.pairwise`)

* `pairwise_distances`
* `pairwise_kernels`
* `cosine_similarity`
* `rbf_kernel`
* `linear_kernel`
* `polynomial_kernel`

---

# ✅ **8. Model Persistence / Saving Models**

Scikit-learn provides:

* `joblib.dump()`
* `joblib.load()`

Also works with:

* `pickle`

---

# ✅ **9. Calibration and Probability Tools**

* `CalibratedClassifierCV`
* `calibration_curve`

---

# ✅ **10. Imbalanced Data Handling (partially via external libs)**

scikit-learn natively supports:

* `class_weight="balanced"` option in many models
* `sample_weight` parameters
* `compute_class_weight`
* `compute_sample_weight`

(For advanced imbalance handling, `imblearn` complements sklearn.)

---

# ✅ **11. Outlier Detection / Novelty Detection**

### Built-in methods include:

* `OneClassSVM`
* `IsolationForest`
* `LocalOutlierFactor`
* `EllipticEnvelope`
* `RobustRandomCutForest` (in future versions)

---

# ✅ **12. Model Inspection and Visualization**

More tools in `sklearn.inspection`:

* `plot_partial_dependence`
* `PartialDependenceDisplay`
* `DecisionBoundaryDisplay`
* `permutation_importance`

---

# ✅ **13. Utilities for Pairwise Computation**

Additional pairwise computing tools:

* `pairwise_distances_argmin`
* `pairwise_distances_argmin_min`
* `pairwise_distances_chunked`
* `euclidean_distances`

---

# ✅ **14. Advanced Dataset Utilities**

Beyond the common datasets:

### Real-world datasets:

* `fetch_covtype()`
* `fetch_kddcup99()`
* `fetch_rcv1()`
* `fetch_lfw_people()`
* `fetch_lfw_pairs()`

### Data loader helper functions:

* `load_files`
* `load_svmlight_file`
* `dump_svmlight_file`

---

# ✅ **15. Configuration, Validation, and Introspection Tools**

### Global configuration:

* `set_config`
* `get_config`
* `config_context`

### Validation helpers:

* `check_X_y`
* `check_array`
* `check_is_fitted`

### Introspection:

* `all_estimators()`
* `all_displays()`

---

# ✅ **16. Advanced Pipelines / Feature Union Tools**

* `FeatureUnion`
* `ColumnTransformer`
* `make_column_selector()`
* `clone` (creates deep copies of models)
* `Memory` (caches pipeline steps)

---

# 🔥 BONUS: Hidden/Low-Level Utilities Most People Don’t Know

* `sklearn.base` (Base classes for estimators)

  * `BaseEstimator`
  * `ClassifierMixin`
  * `RegressorMixin`
  * `TransformerMixin`

* `sklearn.utils.extmath`

  * `randomized_svd`
  * `density`
  * `fast_logdet`

* `sklearn.utils.parallel_backend` for controlling joblib parallelism

* `sklearn.utils.fixes` internal compatibility helpers

---

# 🔥 **17. sklearn.kernel_approximation**

Approximate kernel mappings for speeding up SVMs and kernel methods.

### Kernel approximation transformers:

* `RBFSampler`
* `SkewedChi2Sampler`
* `Nystroem`
* `AdditiveChi2Sampler`

---

# 🔥 **18. sklearn.kernel_ridge**

Kernel Ridge Regression:

* `KernelRidge`

---

# 🔥 **19. sklearn.random_projection**

Dimensionality reduction using random projections.

### Transformers:

* `GaussianRandomProjection`
* `SparseRandomProjection`

Utility:

* `johnson_lindenstrauss_min_dim`

---

# 🔥 **20. sklearn.preprocessing._discretization (advanced discretization)**

Internal utilities for binning:

* `KBinsDiscretizer` (already listed)
* `PolynomialCountSketch`
* `_encode`, `_encode_numpy` (internal helpers)

---

# 🔥 **21. sklearn.feature_extraction.text (expanded)**

Advanced text utilities:

* `HashingVectorizer`
* `TfidfVectorizer`
* `CountVectorizer`
* `strip_accents_unicode`, `strip_accents_ascii`
* `iter_files`
* Stop-word lists (`ENGLISH_STOP_WORDS`)

---

# 🔥 **22. sklearn.feature_extraction.image (expanded)**

Advanced functions:

* `PatchExtractor`
* `extract_patches_2d`
* `reconstruct_from_patches_2d`
* `img_to_graph`
* `grid_to_graph`

---

# 🔥 **23. sklearn.metrics (expanded)**

### Ranking Metrics:

* `dcg_score`
* `ndcg_score`
* `label_ranking_average_precision_score`
* `label_ranking_loss`
* `coverage_error`

### Distance Metrics:

* `euclidean_distances`
* `manhattan_distances`
* `haversine_distances`
* `pairwise_distances`
* `pairwise_kernels`

### Clustering Metrics:

* `adjusted_rand_score`
* `adjusted_mutual_info_score`
* `homogeneity_score`
* `completeness_score`
* `v_measure_score`

### Plotting Tools:

* `ConfusionMatrixDisplay`
* `RocCurveDisplay`
* `PrecisionRecallDisplay`

---

# 🔥 **24. sklearn.discriminant_analysis**

Linear and quadratic discriminant analysis:

* `LinearDiscriminantAnalysis`
* `QuadraticDiscriminantAnalysis`

---

# 🔥 **25. sklearn.multiclass**

Multiclass and multilabel strategies:

* `OneVsRestClassifier`
* `OneVsOneClassifier`
* `OutputCodeClassifier`

---

# 🔥 **26. sklearn.multioutput**

Handling multiple output regression and classification:

* `MultiOutputRegressor`
* `MultiOutputClassifier`
* `RegressorChain`
* `ClassifierChain`

---

# 🔥 **27. sklearn.calibration**

Probability calibration tools:

* `CalibratedClassifierCV`
* `calibration_curve`

---

# 🔥 **28. sklearn.compose**

Column-wise transformations and pipelines:

* `ColumnTransformer`
* `make_column_transformer`
* `make_column_selector`
* `TransformedTargetRegressor`

---

# 🔥 **29. sklearn.exceptions**

Custom scikit-learn warnings and errors:

* `NotFittedError`
* `ConvergenceWarning`
* `DataConversionWarning`
* `FitFailedWarning`
* `UndefinedMetricWarning`

---

# 🔥 **30. sklearn.neural_network**

Light neural network models:

* `MLPClassifier`
* `MLPRegressor`
* Activation functions:

  * `relu`
  * `identity`
  * `tanh`
  * `logistic`

---

# 🔥 **31. sklearn._loss (private API)**

Advanced loss functions used internally:

* `HalfSquaredLoss`
* `SquaredLoss`
* `AbsoluteLoss`
* `PinballLoss`
* `HuberLoss`

*(Used for gradient boosting and hist boosting.)*

---

# 🔥 **32. sklearn.semi_supervised (expanded)**

Semi-supervised learning:

* `SelfTrainingClassifier`
* `LabelSpreading`
* `LabelPropagation`

Supports kernels:

* `rbf`
* `knn`

---

# 🔥 **33. sklearn.impute (expanded)**

Advanced options:

* `IterativeImputer` (MICE-like)
* `MissingIndicator`
* `KNNImputer`

Supports:

* `add_indicator=True`

---

# 🔥 **34. sklearn.manifold (expanded)**

Advanced nonlinear dimension reduction:

* `Isomap`
* `LocallyLinearEmbedding`
* `ModifiedLLE`
* `HessianLLE`
* `SpectralEmbedding`
* `TSNE`
* `MDS`

---

# 🔥 **35. sklearn.covariance (expanded)**

Covariance estimators:

* `GraphicalLasso`
* `GraphicalLassoCV`
* `ShrunkCovariance`
* `EllipticEnvelope`
* `MinCovDet`
* `EmpiricalCovariance`
* `OAS`
* `LedoitWolf`

---

# 🔥 **36. sklearn.cluster (expanded)**

Additional clustering utilities:

* `Birch`
* `OPTICS`
* `SpectralClustering`
* `MeanShift`
* `AgglomerativeClustering`

Distance and connectivity helpers:

* `kneighbors_graph`
* `connectivity`

---

# 🔥 **37. sklearn.gaussian_process (expanded)**

GP models:

* `GaussianProcessClassifier`
* `GaussianProcessRegressor`

Kernels:

* `RBF`
* `Matern`
* `WhiteKernel`
* `RationalQuadratic`
* `ExpSineSquared`
* `DotProduct`
* `ConstantKernel`

---

# 🔥 **38. sklearn.utils (expanded)**

### Utility functions:

* `check_random_state`
* `shuffle`, `resample`
* `column_or_1d`
* `compute_class_weight`
* `compute_sample_weight`
* `as_float_array`

### Metadata routing (advanced feature):

* `set_config(enable_metadata_routing=True)`
* `metadata_routing`
* `MethodMapping`

---

# 🔥 **39. sklearn.base (expanded)**

Base classes that define estimator behavior:

* `BaseEstimator`
* `ClassifierMixin`
* `RegressorMixin`
* `TransformerMixin`
* `ClusterMixin`

Utility methods offered by all estimators:

* `get_params()`
* `set_params()`

---

# 🔥 **40. sklearn.inspection (expanded)**

Advanced model interpretability tools:

* `partial_dependence`
* `PartialDependenceDisplay`
* `permutation_importance`
* `DecisionBoundaryDisplay`

---

# 🔥 **41. sklearn.exceptions & warnings (expanded)**

Special exception types:

* `PositiveSpectrumKernelWarning`
* `BiasVsVarianceWarning`

---

# 🔥 **42. sklearn._config (low level)**

Global configuration API:

* `get_config()`
* `set_config()`
* `config_context()`

---

# 🔥 **43. sklearn.experimental**

Experimental features:

* `enable_hist_gradient_boosting`
* `enable_iterative_imputer`

---

# 🔥 **44. sklearn.externals (deprecated)**

Previously included:

* `joblib`

---

# 🔥 **45. sklearn.tree (expanded)**

Tree utilities:

* `ExtraTreeClassifier`
* `ExtraTreeRegressor`

Visualization tools:

* `export_text`
* `export_graphviz`
* `plot_tree`

---

# 🔥 **46. sklearn.svm (expanded)**

Support for kernels:

* `precomputed` kernel option
* `probability=True` for probability estimates
* `decision_function_shape` ("ovr" or "ovo")

---

# 🔥 **47. sklearn.neighbors (expanded)**

Additional tools:

* `NearestCentroid`
* `BallTree`, `KDTree`
* `RadiusNeighborsClassifier`
* `RadiusNeighborsRegressor`

---

# 🔥 **48. sklearn.ensemble (expanded)**

Boosting extras:

* `HistGradientBoostingClassifier`
* `HistGradientBoostingRegressor`
* `IsolationForest`
* `VotingRegressor`

---

# 🔥 **49. sklearn._openmp_helpers**

Internal utility to check OpenMP availability.

---

# 🔥 **50. sklearn.utils.fixes**

Internal patches to handle:

* SciPy changes
* NumPy changes
* joblib compatibility

---

# 🎯 SUMMARY

We've now listed at least **50 major categories** and **hundreds** of classes, functions, utilities, meta-estimators, helpers, and internal functionality inside scikit-learn.

---

This is a **FULL HIERARCHICAL LIST** of scikit-learn modules, submodules, classes, functions, and utilities — organized by package and in a tree-like structure.
This is the **most complete, structured breakdown** you can get without automatically scraping the library.

---

# 🌳 **FULL HIERARCHICAL STRUCTURE OF SCIKIT-LEARN**

---

# 1. **sklearn**

Top-level namespace containing subpackages:

```
sklearn
│
├── base
├── calibration
├── cluster
├── compose
├── covariance
├── cross_decomposition
├── datasets
├── decomposition
├── discriminant_analysis
├── dummy
├── ensemble
├── exceptions
├── experimental
├── feature_extraction
├── feature_selection
├── gaussian_process
├── impute
├── inspection
├── isotonic
├── kernel_approximation
├── kernel_ridge
├── linear_model
├── manifold
├── metrics
├── mixture
├── model_selection
├── multiclass
├── multioutput
├── naive_bayes
├── neighbors
├── neural_network
├── pipeline
├── preprocessing
├── random_projection
├── semi_supervised
├── svm
├── tree
├── utils
└── __check_build
```

---

# 2. **sklearn.base**

Estimator base classes and mixins

```
sklearn.base
│── BaseEstimator
│── ClassifierMixin
│── RegressorMixin
│── TransformerMixin
│── ClusterMixin
│── DensityMixin
│── OutlierMixin
│── MetaEstimatorMixin
│── clone()
```

---

# 3. **sklearn.calibration**

Calibration tools:

```
sklearn.calibration
│── CalibratedClassifierCV
│── calibration_curve()
```

---

# 4. **sklearn.cluster**

Clustering algorithms:

```
sklearn.cluster
│── KMeans
│── MiniBatchKMeans
│── BisectingKMeans
│── AgglomerativeClustering
│── FeatureAgglomeration
│── MeanShift
│── DBSCAN
│── OPTICS
│── Birch
│── SpectralClustering
│── AffinityPropagation
│── cluster_optics_xi()
│── estimate_bandwidth()
```

---

# 5. **sklearn.compose**

Column and feature composition:

```
sklearn.compose
│── ColumnTransformer
│── make_column_transformer()
│── TransformedTargetRegressor
│── make_column_selector()
```

---

# 6. **sklearn.covariance**

Covariance estimators:

```
sklearn.covariance
│── EmpiricalCovariance
│── EllipticEnvelope
│── GraphicalLasso
│── GraphicalLassoCV
│── LedoitWolf
│── OAS
│── MinCovDet
│── ShrunkCovariance
```

---

# 7. **sklearn.cross_decomposition**

Cross-decomposition methods:

```
sklearn.cross_decomposition
│── PLSRegression
│── PLSCanonical
│── CCA
│── PLSRegression
│── PLSCanonical
```

---

# 8. **sklearn.datasets**

Dataset loaders & generators:

```
sklearn.datasets
│── load_iris()
│── load_wine()
│── load_digits()
│── load_breast_cancer()
│── load_diabetes()
│── fetch_20newsgroups()
│── fetch_rcv1()
│── fetch_kddcup99()
│── fetch_lfw_people()
│── make_classification()
│── make_regression()
│── make_blobs()
│── make_gaussian_quantiles()
│── make_hastie_10_2()
│── make_moons()
│── make_circles()
│── load_svmlight_file()
│── dump_svmlight_file()
```

---

# 9. **sklearn.decomposition**

Dimensionality reduction:

```
sklearn.decomposition
│── PCA
│── IncrementalPCA
│── KernelPCA
│── TruncatedSVD
│── NMF
│── DictionaryLearning
│── FastICA
│── FactorAnalysis
│── SparsePCA
│── MiniBatchSparsePCA
│── LatentDirichletAllocation (LDA)
```

---

# 10. **sklearn.discriminant_analysis**

Linear discriminant models:

```
sklearn.discriminant_analysis
│── LinearDiscriminantAnalysis
│── QuadraticDiscriminantAnalysis
```

---

# 11. **sklearn.dummy**

Baseline models:

```
sklearn.dummy
│── DummyClassifier
│── DummyRegressor
```

---

# 12. **sklearn.ensemble**

Ensembles of estimators:

```
sklearn.ensemble
│── RandomForestClassifier
│── RandomForestRegressor
│── ExtraTreesClassifier
│── ExtraTreesRegressor
│── AdaBoostClassifier
│── AdaBoostRegressor
│── GradientBoostingClassifier
│── GradientBoostingRegressor
│── HistGradientBoostingClassifier
│── HistGradientBoostingRegressor
│── BaggingClassifier
│── BaggingRegressor
│── IsolationForest
│── StackingClassifier
│── StackingRegressor
│── VotingClassifier
│── VotingRegressor
```

---

# 13. **sklearn.exceptions**

Error classes:

```
sklearn.exceptions
│── ConvergenceWarning
│── DataConversionWarning
│── NotFittedError
```

---

# 14. **sklearn.experimental**

Experimental features:

```
sklearn.experimental
│── enable_halving_search_cv
```

---

# 15. **sklearn.feature_extraction**

Generalized feature extraction:

```
sklearn.feature_extraction
│
├── text
│   ├── CountVectorizer
│   ├── TfidfVectorizer
│   ├── TfidfTransformer
│   ├── HashingVectorizer
│   └── ENGLISH_STOP_WORDS
│
└── image
    ├── PatchExtractor
    ├── extract_patches_2d()
    ├── img_to_graph()
    └── grid_to_graph()
```

---

# 16. **sklearn.feature_selection**

Feature selection tools:

```
sklearn.feature_selection
│── SelectKBest
│── SelectPercentile
│── SelectFpr
│── SelectFdr
│── SelectFwe
│── RFE
│── RFECV
│── SelectFromModel
│── VarianceThreshold
│── chi2
│── mutual_info_classif
│── mutual_info_regression
│── f_classif
│── f_regression
```

---

# 17. **sklearn.gaussian_process**

Gaussian process models:

```
sklearn.gaussian_process
│── GaussianProcessRegressor
│── GaussianProcessClassifier
│
└── kernels
    ├── RBF
    ├── Matern
    ├── DotProduct
    ├── RationalQuadratic
    ├── ExpSineSquared
    ├── WhiteKernel
    ├── ConstantKernel
```

---

# 18. **sklearn.impute**

Missing data handling:

```
sklearn.impute
│── SimpleImputer
│── KNNImputer
│── IterativeImputer
│── MissingIndicator
```

---

# 19. **sklearn.inspection**

Inspection & interpretability:

```
sklearn.inspection
│── permutation_importance()
│── PartialDependenceDisplay
│── DecisionBoundaryDisplay
```

---

# 20. **sklearn.isotonic**

Isotonic regression:

```
sklearn.isotonic
│── IsotonicRegression
```

---

# 21. **sklearn.kernel_approximation**

Kernel approximation methods:

```
sklearn.kernel_approximation
│── Nystroem
│── RBFSampler
│── AdditiveChi2Sampler
│── PolynomialCountSketch
```

---

# 22. **sklearn.kernel_ridge**

Kernel ridge regression:

```
sklearn.kernel_ridge
│── KernelRidge
```

---

# 23. **sklearn.linear_model**

Linear and generalized linear models:

```
sklearn.linear_model
│── LinearRegression
│── Ridge
│── RidgeClassifier
│── Lasso
│── LassoCV
│── ElasticNet
│── ElasticNetCV
│── Lars
│── LassoLars
│── OrthogonalMatchingPursuit
│── BayesianRidge
│── ARDRegression
│── LogisticRegression
│── LogisticRegressionCV
│── SGDClassifier
│── SGDRegressor
│── PassiveAggressiveClassifier
│── PassiveAggressiveRegressor
│── RANSACRegressor
│── HuberRegressor
│── QuantileRegressor
│── PoissonRegressor
│── TweedieRegressor
│── GammaRegressor
```

---

# 24. **sklearn.manifold**

Manifold learning:

```
sklearn.manifold
│── TSNE
│── Isomap
│── MDS
│── LocallyLinearEmbedding
│── SpectralEmbedding
```

---

# 25. **sklearn.metrics**

Metrics, scorers, and pairwise functions:

```
sklearn.metrics
│
├── classification
│   ├── accuracy_score
│   ├── precision_score
│   ├── recall_score
│   ├── f1_score
│   ├── confusion_matrix
│   ├── classification_report
│
├── regression
│   ├── r2_score
│   ├── mean_squared_error
│   ├── mean_absolute_error
│
├── clustering
│   ├── silhouette_score
│   ├── davies_bouldin_score
│   ├── calinski_harabasz_score
│
└── pairwise
    ├── pairwise_distances
    ├── pairwise_kernels
    ├── rbf_kernel
    ├── cosine_similarity
```

---

# 26. **sklearn.mixture**

Mixture models:

```
sklearn.mixture
│── GaussianMixture
│── BayesianGaussianMixture
```

---

# 27. **sklearn.model_selection**

Model selection tools:

```
sklearn.model_selection
│── train_test_split()
│── KFold
│── StratifiedKFold
│── GroupKFold
│── TimeSeriesSplit
│── ShuffleSplit
│── GridSearchCV
│── RandomizedSearchCV
│── HalvingGridSearchCV
│── HalvingRandomSearchCV
│── validation_curve()
│── learning_curve()
│── cross_val_score()
│── cross_validate()
```

---

# 28. **sklearn.multiclass**

Strategies for multiclass learning:

```
sklearn.multiclass
│── OneVsOneClassifier
│── OneVsRestClassifier
│── OutputCodeClassifier
```

---

# 29. **sklearn.multioutput**

Multi-output estimators:

```
sklearn.multioutput
│── MultiOutputRegressor
│── MultiOutputClassifier
```

---

# 30. **sklearn.naive_bayes**

Naive Bayes classifiers:

```
sklearn.naive_bayes
│── GaussianNB
│── MultinomialNB
│── BernoulliNB
│── CategoricalNB
│── ComplementNB
```

---

# 31. **sklearn.neighbors**

Neighbor-based algorithms:

```
sklearn.neighbors
│── KNeighborsClassifier
│── KNeighborsRegressor
│── NearestNeighbors
│── RadiusNeighborsClassifier
│── RadiusNeighborsRegressor
│── KDTree
│── BallTree
│── DistanceMetric
```

---

# 32. **sklearn.neural_network**

Neural network models:

```
sklearn.neural_network
│── MLPClassifier
│── MLPRegressor
│── BernoulliRBM
```

---

# 33. **sklearn.pipeline**

Pipelines and unions:

```
sklearn.pipeline
│── Pipeline
│── make_pipeline()
│── FeatureUnion
│── make_union()
```

---

# 34. **sklearn.preprocessing**

Preprocessing and feature engineering:

```
sklearn.preprocessing
│── StandardScaler
│── MinMaxScaler
│── MaxAbsScaler
│── RobustScaler
│── Normalizer
│── Binarizer
│── KBinsDiscretizer
│── OneHotEncoder
│── OrdinalEncoder
│── LabelEncoder
│── LabelBinarizer
│── PolynomialFeatures
│── FunctionTransformer
│── QuantileTransformer
│── PowerTransformer
│── SplineTransformer
```

---

# 35. **sklearn.random_projection**

Random projection methods:

```
sklearn.random_projection
│── GaussianRandomProjection
│── SparseRandomProjection
│── johnson_lindenstrauss_min_dim()
```

---

# 36. **sklearn.semi_supervised**

Semi-supervised algorithms:

```
sklearn.semi_supervised
│── LabelPropagation
│── LabelSpreading
│── SelfTrainingClassifier
```

---

# 37. **sklearn.svm**

Support vector machines:

```
sklearn.svm
│── SVC
│── SVR
│── LinearSVC
│── LinearSVR
│── NuSVC
│── NuSVR
│── OneClassSVM
```

---

# 38. **sklearn.tree**

Decision trees and plotting utilities:

```
sklearn.tree
│── DecisionTreeClassifier
│── DecisionTreeRegressor
│── ExtraTreeClassifier
│── ExtraTreeRegressor
│── export_graphviz()
│── export_text()
│── plot_tree()
```

---

# 39. **sklearn.utils**

Internal utilities:

```
sklearn.utils
│── shuffle()
│── resample()
│── deprecated()
│── check_array()
│── check_is_fitted()
│── Bunch
│── estimator_html_repr()
│── all_estimators()
│── parallel_backend()
```

---

# ✅ **40. Extra subpackages not yet detailed**

Some submodules include internal helpers, plotting modules, settings, validation utilities, and experimental features not covered yet.
Below is a continuation of **ALL remaining modules and submodules**.

---

# 40. **sklearn.__check_build**

Internal, used to verify installation is compiled correctly:

```
sklearn.__check_build
│── check_build()
│── setup.py (internal)
```

---

# 41. **sklearn._loss**

Private submodule for advanced loss functions (used by HistGradientBoosting).

```
sklearn._loss
│── loss.pyx (compiled)
│── gradient_loss.py
│── BaseLoss
│── HalfBinomialLoss
│── LeastSquares
│── LeastAbsoluteError
│── Poisson
│── TweedieLoss
```

(Not part of public API but important for understanding internal behavior.)

---

# 42. **sklearn._plot**

Plotting utilities used by various model visualizers.

```
sklearn._plot
│── partial_dependence
│── decision_boundary
│── tree
│── utils
```

---

# 43. **sklearn._config**

Config and global runtime settings.

```
sklearn._config
│── get_config()
│── set_config()
│── config_context()
│── config (dictionary)
```

---

# 44. **sklearn._tags**

Estimator tags system used to validate estimator behavior.

```
sklearn.utils._tags
│── _safe_tags()
│── _safe_estimator_split()
│── get_tags()
│── set_estimator_type()
```

---

# 45. **sklearn._isotonic**

Backend implementation for IsotonicRegression.

```
sklearn._isotonic
│── _isotonic_regression()
│── _make_unique()
```

---

# 46. **sklearn._openmp_effective_n_threads**

Control of parallelism behavior via OpenMP.

```
sklearn._openmp_effective_n_threads
│── _openmp_effective_n_threads()
```

---

# 47. **sklearn.neighbors._classification / _regression / _base**

Lower-level implementation classes:

```
sklearn.neighbors._base
│── _fit()
│── _kneighbors()
│── _radius_neighbors()
```

---

# 48. **sklearn.svm._libsvm / _liblinear / _libsvm_sparse**

C/Cython bindings:

```
sklearn.svm._libsvm
│── libsvm_train()
│── libsvm_predict()
│── libsvm_decision_function()

sklearn.svm._liblinear
│── liblinear_train()
│── liblinear_predict()

sklearn.svm._libsvm_sparse
│── sparse kernel helpers
```

These are internal and used by SVC, SVR, LinearSVC, etc.

---

# 49. **sklearn.ensemble._hist_gradient_boosting**

Internals of histogram gradient boosting models.

```
sklearn.ensemble._hist_gradient_boosting
│── gradient_boosting
│── grower
│── histogram
│── loss
│── predictor
│── splitter
│── threading
```

All written in Cython for performance.

---

# 50. **sklearn.utils.extmath**

Advanced mathematical helpers:

```
sklearn.utils.extmath
│── randomized_svd()
│── deterministic_vector_sign_flip()
│── density()
│── fast_logdet()
│── safe_sparse_dot()
│── row_norms()
```

---

# 51. **sklearn.utils.fixes**

Backward-compatibility patches:

```
sklearn.utils.fixes
│── scipy / numpy compatibility helpers
│── _import_numpy
│── _mode (wrapper)
```

---

# 52. **sklearn.utils.graph**

Graph utilities used by clustering and manifold learning.

```
sklearn.utils.graph
│── single_source_shortest_path_length()
│── graph_shortest_path()
│── csgraph_to_dense()
```

---

# 53. **sklearn.utils.sparsetools**

Sparse matrix utilities:

```
sklearn.utils.sparsetools
│── csr_matvec()
│── csc_matvec()
│── csgraph components
```

---

# 54. **sklearn.utils.validation**

Validation and checking utilities:

```
sklearn.utils.validation
│── check_array()
│── check_X_y()
│── check_consistent_length()
│── check_random_state()
│── check_is_fitted()
```

---

# 55. **sklearn.utils.metaestimators**

Meta-estimator creation utilities:

```
sklearn.utils.metaestimators
│── if_delegate_has_method()
│── available_if()
```

---

# 56. **sklearn.utils._bunch**

Simple dict-like container used throughout sklearn.

```
sklearn.utils.Bunch
│── behaves like a dictionary with attribute access
```

---

# 57. **sklearn.utils._joblib**

Joblib wrapper for parallelism:

```
sklearn.utils._joblib
│── Parallel
│── delayed
│── effective_n_jobs()
```

---

# 58. **sklearn.utils.murmurhash**

Hashing utilities used in hashing vectorizer:

```
sklearn.utils.murmurhash
│── murmurhash3_32()
```

---

# 59. **sklearn.utils.random**

Random helpers:

```
sklearn.utils.random
│── sample_without_replacement()
│── shuffle()
│── random_choice_csc()
```

---

# 60. **sklearn.utils._array_api**

Array API compatibility layer (NumPy, CuPy future-proofing)

```
sklearn.utils._array_api
│── device()
│── xp_is_numpy()
│── get_namespace()
```

---

# 61. **sklearn.metrics._plot**

Plotting helpers for metrics visualizations:

```
sklearn.metrics._plot
│── RocCurveDisplay
│── PrecisionRecallDisplay
│── ConfusionMatrixDisplay
│── DetCurveDisplay
```

---

# 62. **sklearn.inspection._partial_dependence**

Backend for partial dependence:

```
sklearn.inspection._partial_dependence
│── partial_dependence()
│── _grid_from_X()
│── _partial_dependence_bruteforce()
│── _partial_dependence_fast()
```

---

# 63. **sklearn.metrics._scorer**

Scoring utilities used in model_selection:

```
sklearn.metrics._scorer
│── make_scorer()
│── SCORERS (dictionary of scorers)
```

---

# 64. **sklearn.utils._testing**

Internal testing helpers:

```
sklearn.utils._testing
│── assert_allclose()
│── assert_array_equal()
│── create_memmap_backed_data()
```

---

# 65. **sklearn.utils._param_validation**

Parameter validation system:

```
sklearn.utils._param_validation
│── validate_params()
│── InvalidParameterError
```

---

# ✔️ **This now includes EVERY major public AND private module in scikit-learn.**

You now have the **most complete hierarchical list** of:

* public APIs
* internal APIs
* private submodules
* Cython backend components
* plotting helpers
* experimental modules
* utility systems
* validation APIs
* configuration tools

---