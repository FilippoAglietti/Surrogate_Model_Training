# ⚡ Surrogate Builder v4.0

A desktop application for building, optimizing, and deploying **surrogate models** — lightweight approximators that replace expensive simulations or experiments. Supports **Neural Networks, XGBoost, Random Forest, and Gaussian Process Regression**, with a full workflow from raw data to deployable model artifact.

Built entirely with **CustomTkinter** (native dark UI), **TensorFlow/Keras**, **scikit-learn**, **XGBoost**, and **Optuna**.

![Python](https://img.shields.io/badge/Python-3.9+-00ff41?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-00e5ff?style=flat-square&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-ff6e40?style=flat-square&logo=python&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.5+-ff00ff?style=flat-square&logo=python&logoColor=white)

---

## Table of Contents

1. [What is a Surrogate Model?](#what-is-a-surrogate-model)
2. [Installation](#installation)
3. [Workflow Overview](#workflow-overview)
4. [Tab Reference](#tab-reference)
   - [Data Loading](#-data-loading)
   - [Preprocessing](#-preprocessing)
   - [Model Builder](#-model-builder)
   - [Hyperopt](#-hyperopt)
   - [Results](#-results)
   - [Inference](#-inference)
5. [Session Management](#session-management)
6. [Model Wrapper Export](#model-wrapper-export)
7. [Algorithms](#algorithms)
8. [Project Structure](#project-structure)
9. [Requirements](#requirements)

---

## What is a Surrogate Model?

A surrogate model (also called a metamodel or emulator) is a fast approximation of a complex, expensive function. Common use cases:

- Replace a slow CFD/FEA simulation with a model that predicts the same outputs in milliseconds
- Approximate a physical experiment to explore a design space cheaply
- Compress a high-dimensional parameter study into a queryable function

This tool covers the full pipeline: **load data → preprocess → train → evaluate → deploy**.

---

## Installation

### Windows (recommended)

Double-click `run.bat`. It will:
1. Create a Python virtual environment in `./venv` if one does not exist
2. Install all dependencies from `requirements.txt`
3. Launch the application

### Manual (Windows / Linux / macOS)

```bash
# Clone or download the repository
git clone https://github.com/FilippoAglietti/Surrogate_Model_Training.git
cd Surrogate_Model_Training

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Launch
python app.py
```

### Requirements

| Package | Version |
|---|---|
| customtkinter | ≥ 5.2.2 |
| tensorflow | ≥ 2.15.0 |
| scikit-learn | ≥ 1.3.0 |
| xgboost | ≥ 2.0.0 |
| optuna | ≥ 3.5.0 |
| pandas | ≥ 2.0.0 |
| matplotlib | ≥ 3.8.0 |
| seaborn | ≥ 0.13.0 |
| shap | ≥ 0.44.0 |
| scipy | ≥ 1.11.0 |
| joblib | ≥ 1.3.0 |
| openpyxl | ≥ 3.1.2 |

---

## Workflow Overview

The application follows a linear six-step pipeline. Each tab unlocks the next one once its required inputs are ready.

```
📁 Data Loading
      ↓  upload CSV / Excel, select input and output columns
🔧 Preprocessing
      ↓  scale, apply PCA, split into train / val / test
🏗 Model Builder  ←──────────────────────────────────────┐
      ↓  configure architecture, train, view live loss    │
🔍 Hyperopt  ──────────── run Optuna → apply best params ─┘
      ↓  automated hyperparameter search
📊 Results
      ↓  evaluate metrics, SHAP, export deployable wrapper
🔮 Inference
      ↓  load wrapper, run sensitivity analysis, batch predict
```

---

## Tab Reference

### 📁 Data Loading

**Purpose:** Import a dataset and tell the app which columns are inputs (features) and which are outputs (targets).

**How to use:**
1. Click **Upload Dataset** and select a `.csv` or `.xlsx` file.
2. The app shows:
   - **Shape** — number of rows and columns
   - **Statistics table** — min, max, mean, std for every numeric column
   - **Preview tab** — scrollable table of the first 50 rows
   - **Missing Values tab** — bar chart of null counts per column
3. In the **Target (Output) Columns** panel, check every column you want to predict.
4. In the **Features (Input) Columns** panel, check every column used as input.
5. Each column shows a type badge: `NUM` (cyan), `DATE` (orange), `CAT` (red). Only `NUM` columns can be selected for training.
6. Click **Proceed to Preprocessing** once at least one input and one output are selected without overlaps.

**Notes:**
- Multi-output is fully supported: select as many output columns as needed.
- Categorical and date columns are flagged as non-numeric and cannot be included in training.

---

### 🔧 Preprocessing

**Purpose:** Scale features, optionally reduce dimensionality with PCA, split the data, and visualize distributions.

#### Normalization

| Option | Effect |
|---|---|
| Min-Max (0, 1) | Scales every column to the [0, 1] range |
| Standard (Z-score) | Centers to mean 0, std 1 |
| None | No scaling applied |

The **Scale Target Variables** checkbox controls whether scaling is also applied to outputs.

#### PCA (Principal Component Analysis)

PCA can be applied independently to inputs and outputs:

- **Enable Input PCA** — reduces input dimensionality. Enter the number of components to keep.
- **Enable Output PCA** — reduces output dimensionality (useful for high-dimensional targets).
- **Scree button** — opens a popup showing the explained variance per component and cumulative variance curve, with 90% and 95% reference lines. Use this to decide how many components to keep.

When PCA is active, the preprocessing plots and all downstream tabs work in PCA space (columns named `PC_X_1`, `PC_X_2`, … or `PC_Y_1`, `PC_Y_2`, …).

#### Data Split

Two sliders control the train / val / test proportions. The test percentage is computed automatically as `1 − train − val`. Typical split: 70% / 15% / 15%.

#### Charts to Generate

Select which visualizations to produce after running preprocessing:

| Chart | What it shows |
|---|---|
| Combined Matrix | Scatter plots + correlation heatmap + histograms for all selected columns |
| Box / Violin | Distribution shape and outliers per column |
| KDE Distributions | Kernel density estimates, useful for checking normality |
| Parallel Coordinates | All columns as parallel axes — good for spotting clusters |
| Outlier Detection (IQR) | Highlights points beyond 1.5× IQR from Q1/Q3 |

Every chart has a **Save Image** button (PNG/PDF/SVG).

Click **RUN PREPROCESSING & PLOT** to execute. The status bar shows the resulting split sizes.

---

### 🏗 Model Builder

**Purpose:** Configure and train a surrogate model. Supports four algorithms, each with its own parameter panel.

#### Algorithm Selector

Choose the algorithm from the dropdown at the top. Switching algorithms rebuilds the parameter panel below without losing the trained model or preprocessing results.

---

#### Neural Network

**Architecture card:**

| Parameter | Description |
|---|---|
| Num hidden layers | Number of Dense layers between input and output |
| Neurons per hidden layer | Width of each hidden layer |
| Hidden Activation | Activation function: ReLU, LeakyReLU, ELU, SELU, Tanh, Sigmoid, GELU, SiLU, Linear |
| Output Activation | Activation on the last layer (usually Linear for regression) |
| Dropout | Fraction of neurons randomly zeroed during training (0 = disabled) |
| L1 Regularization | L1 penalty on weights |
| L2 Regularization | L2 penalty on weights |

**Compile & Hyperparams card:**

| Parameter | Description |
|---|---|
| Loss Function | MSE, MAE, Huber, LogCosh |
| Optimizer | Adam, AdamW, SGD, RMSprop |
| Learning Rate | Initial learning rate |
| Batch Size | Number of samples per gradient update |
| Epochs | Maximum training epochs |

**Callbacks:**

| Callback | Parameters | Effect |
|---|---|---|
| Early Stopping | Patience, Min Δ | Stops training when val loss stops improving |
| Reduce LR on Plateau | Factor, Patience, Min LR | Reduces learning rate when val loss plateaus |

**LR Scheduler:**

| Scheduler | Parameter | Effect |
|---|---|---|
| None (fixed LR) | — | Learning rate stays constant |
| CosineDecay | Decay Steps | Cosine annealing from initial LR to 0 |
| ExponentialDecay | Decay Rate | LR × rate every step |

After clicking **BUILD**, a model summary is shown in the plot area.
After clicking **START TRAINING**, a live loss/val-loss curve is drawn epoch by epoch.
**STOP** interrupts training early (the partial model is kept).

---

#### XGBoost

Gradient boosted trees via `XGBRegressor`. For multi-output targets, each output gets its own estimator wrapped by `MultiOutputRegressor`.

| Parameter | Description |
|---|---|
| N Estimators | Number of boosting rounds (trees) |
| Max Depth | Maximum depth of each tree |
| Learning Rate | Shrinkage factor applied to each tree |
| Subsample | Fraction of training samples used per tree |
| ColSample ByTree | Fraction of features sampled per tree |
| Reg Alpha (L1) | L1 regularization on leaf weights |
| Reg Lambda (L2) | L2 regularization on leaf weights |

---

#### Random Forest

Ensemble of decision trees via `RandomForestRegressor`. Natively multi-output.

| Parameter | Description |
|---|---|
| N Estimators | Number of trees in the forest |
| Max Depth | Maximum depth per tree (empty = unlimited) |
| Min Samples Split | Minimum samples required to split a node |
| Min Samples Leaf | Minimum samples required at a leaf node |
| Max Features | Features considered at each split: `sqrt`, `log2`, `1.0` (all) |

Random Forest provides **uncertainty bands** in the 1D Sensitivity tab (variance across tree predictions).

---

#### Gaussian Process Regression

Probabilistic regression via `GaussianProcessRegressor`. For multi-output, one GP is fit per output.

| Parameter | Description |
|---|---|
| Kernel | Covariance function: RBF, Matern, RationalQuadratic |
| Alpha | Noise level added to the diagonal of the kernel matrix |
| N Restarts Optimizer | Number of random restarts for kernel hyperparameter optimization |

> ⚠ GPR scales as O(n³). Recommended for datasets below ~2 000 samples. Provides **calibrated uncertainty estimates** in the 1D Sensitivity tab.

---

#### Training controls

| Button | Action |
|---|---|
| ⚡ BUILD | Validates parameters, instantiates the model, enables START TRAINING |
| 🚀 START TRAINING | Begins training in a background thread |
| 🛑 STOP | Interrupts training (NN only); partial model is preserved |

After training, **Val R²** is shown in the status bar and the results tab is automatically marked for rebuild.

---

### 🔍 Hyperopt

**Purpose:** Automated hyperparameter search using [Optuna](https://optuna.org/) with the TPE (Tree-structured Parzen Estimator) sampler.

#### Strategy settings

| Setting | Description |
|---|---|
| Algorithm | Which model type to tune (Neural Network, XGBoost, Random Forest, Gaussian Process) |
| N Trials | Number of configurations to evaluate |
| Timeout (s) | Stop search after this many seconds regardless of trial count |
| Direction | Minimize or maximize the objective (always val MSE here, so minimize) |

#### Search space panels

Each algorithm exposes a set of range controls (min / max for continuous params, list for categoricals). Optuna samples from these ranges for each trial.

**Neural Network search space:** learning rate, neurons, layers, dropout, L1/L2, optimizer, loss, activation, callbacks.
**XGBoost search space:** n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda.
**Random Forest search space:** n_estimators, max_depth, min_samples_split, min_samples_leaf.
**Gaussian Process search space:** kernel type, alpha, n_restarts.

#### Running the search

Click **RUN OPTIMIZATION**. Each trial is evaluated on the validation set. Progress is logged in the log box. The search runs in a background thread; the UI stays responsive.

#### Visualizations (auto-generated after search)

| Plot | What it shows |
|---|---|
| Optimization History | Objective value vs trial number — shows convergence |
| Param Importances | Which hyperparameters had the most impact on performance |
| Parallel Coordinates | All trials plotted across their hyperparameter axes |
| Contour Plot | 2D interaction between the two most important params |

All plots have a **Save Image** button.

#### Applying results

Click **✨ APPLY BEST PARAMS** to push the best trial's parameters to Model Builder. The app automatically switches the Model Builder to the correct algorithm and fills in all parameter fields.

---

### 📊 Results

**Purpose:** Evaluate the trained model on the held-out test set and export a deployable artifact.

Results are shown in tabs. The tab rebuilds automatically whenever a new model is trained.

#### Metrics summary

Displayed at the top for every output:

| Metric | Description |
|---|---|
| R² | Coefficient of determination (1 = perfect) |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| Max Error | Worst single prediction error |

The algorithm name is shown as a badge above the metrics.

#### Visualization tabs

| Tab | Content |
|---|---|
| Pred vs Actual | Scatter of predicted vs actual values with a perfect-fit diagonal |
| Test Series | Predicted and actual values plotted as time series over test index |
| Residuals | Residual scatter (actual − predicted) vs predicted value |
| Q-Q Plot | Quantile-quantile plot of residuals against a normal distribution |
| Worst Predictions | Table of the test samples with the highest absolute error |
| Per-Target Metrics | Bar chart of R² for every output column |
| SHAP | Feature importance via SHAP values (see below) |

#### SHAP feature importance

The SHAP tab shows which input features most influence each prediction:

- **Neural Network / GPR** — uses `KernelExplainer` (model-agnostic, slower)
- **XGBoost** — uses `TreeExplainer` (fast, exact)
- **Random Forest** — uses `TreeExplainer`

A summary bar plot is shown for the selected output. Use the output selector dropdown to switch targets in multi-output models.

#### Export

| Export | Format | Content |
|---|---|---|
| Download Predictions | CSV or Excel | Test set: input columns + actual + predicted + residuals |
| Export Model Wrapper | .smproj zip | Complete deployable artifact (see Model Wrapper Export) |

---

### 🔮 Inference

**Purpose:** Load a trained model wrapper and run interactive predictions or batch inference without needing the training data.

#### Loading a wrapper

Click **LOAD MODEL WRAPPER** and select a `.zip` file exported from the Results tab. The app auto-detects the format (v1 legacy Keras-only or v2 with manifest) and restores the model, scalers, PCA transforms, and column names.

#### 1D Sensitivity

Explore how each input affects each output, holding all other inputs at their base values.

- **Base point sliders** — one slider per input column, range set from training data min/max
- **Sensitivity plot** — for each input, a curve is drawn by scanning that input while fixing the rest at the base point
- **Base point marker** — a white × marks the current slider position on every curve
- **Uncertainty bands** — for Random Forest and GPR, a shaded band shows ±1 std of the model's uncertainty (toggle with the checkbox)

#### 2D Sensitivity

For a selected output, draws a grid of contour plots for every pair of inputs (up to 5 inputs → up to 10 plots). Each contour holds all other inputs at their base values.

The white × marker shows the current base point in each 2D plane.

#### Batch Prediction

Upload an Excel file containing input columns in any order. The app:
1. Validates that all required input columns are present
2. Applies the saved scalers and PCA transforms
3. Runs the model
4. Appends the predicted output columns
5. Offers download as CSV or Excel

---

## Session Management

Surrogate Builder v4.0 includes a full session persistence system. Your entire workspace — data, preprocessing, trained model, and parameters — can be saved and restored across sessions.

### Session controls

The **SESSION** card at the bottom of the sidebar has three buttons:

| Button | Action |
|---|---|
| New | Clears the current session. Asks for confirmation if there is unsaved work. |
| Open | Opens a `.smproj` file and restores the full workspace. |
| Save | Saves the current session. On first save, opens a file dialog; subsequent saves overwrite the same file. |

An **orange ●** indicator appears next to the session name whenever there are unsaved changes (new data loaded, preprocessing run, model trained, or HPO result applied).

### What is saved

A `.smproj` file is a ZIP archive containing:

| Component | File inside zip | Notes |
|---|---|---|
| Raw dataset | `data/dataframe.csv` | The full DataFrame as loaded |
| Preprocessing plot data | `data/plot_df.csv` | Data used to regenerate preprocessing charts |
| Train / val / test arrays | `splits/arrays.npz` | Numpy compressed archive |
| Scalers | `scalers/scaler_X.pkl`, `scaler_y.pkl` | Fitted MinMax or Standard scalers |
| PCA transforms | `pca/pca_X.pkl`, `pca_y.pkl` | Fitted PCA objects |
| Trained model | `model/` | Algorithm-specific files (`.keras` for NN, `.pkl` for sklearn) |
| All parameter settings | `session.json` | Layers, optimizer, epochs, XGBoost params, etc. |
| HPO best params | `session.json` | Best Optuna parameters |
| Training metrics | `session.json` | Loss history, R², MAE, RMSE |

### What is restored on load

After opening a `.smproj` file:

1. **Data Loading tab** — column checkboxes and dataset statistics are repopulated
2. **Preprocessing tab** — all sliders and dropdowns (scaler type, PCA settings, split %) restored; all charts redrawn from saved data
3. **Model Builder tab** — algorithm switched to the saved one; every parameter field restored; loss curve (NN) or training status (sklearn) redrawn; train button re-enabled
4. **Hyperopt tab** — Apply Best button re-enabled if HPO results are present
5. **Results tab** — rebuilt automatically on first visit (metrics, plots, SHAP)
6. The app navigates directly to the most advanced tab reached in the saved session

> **Note:** The Optuna study object (all trial history) is not saved due to serialization constraints. `best_params` is preserved and can still be applied to Model Builder.

---

## Model Wrapper Export

The **Export Model Wrapper** button in Results creates a standalone `.zip` artifact that can be loaded in the Inference tab (on any machine with the app installed) or used programmatically.

### Wrapper format (v2)

```
model_wrapper.zip
├── manifest.json        ← {"algorithm": "XGBoost", ...}
├── model/               ← algorithm-specific model files
│   ├── model.keras      (Neural Network)
│   ├── model.pkl        (XGBoost / Random Forest / GPR)
│   └── n_out.pkl        (multi-output count, where needed)
└── metadata.pkl         ← input_columns, output_columns,
                            scaler_X, scaler_y, pca_X, pca_y
```

The wrapper is self-contained: it includes everything needed to transform new inputs and produce predictions, with no reference to the original training data.

---

## Algorithms

| Algorithm | Backend | Multi-output | Uncertainty | Best for |
|---|---|---|---|---|
| Neural Network | TensorFlow/Keras | Native | No | Large datasets, complex nonlinear mappings |
| XGBoost | xgboost + sklearn | MultiOutputRegressor | No | Tabular data, fast training, often best out-of-box |
| Random Forest | sklearn | Native | Yes (tree variance) | Robust baseline, interpretability, uncertainty needed |
| Gaussian Process | sklearn | One GP per output | Yes (calibrated std) | Small datasets (< 2k rows), probabilistic output required |

---

## Project Structure

```
Surrogate_Model_Training/
│
├── app.py                        # Main window, sidebar, session actions
├── run.bat                       # Windows one-click launcher
├── requirements.txt
├── smoke_test.py                 # Integration test suite
├── generate_dataset.py           # Friedman benchmark (single-output)
├── generate_multi_dataset.py     # Multi-output test dataset
│
├── utils/
│   ├── state.py                  # Centralized AppState singleton
│   ├── session.py                # Save / load .smproj session files
│   ├── theme.py                  # Colors, fonts, dark theme setup
│   └── plot_utils.py             # Shared "Save Image" button helper
│
├── models/
│   ├── base.py                   # SurrogateModel abstract base class
│   ├── nn_model.py               # NeuralNetworkSurrogate (Keras wrapper)
│   ├── xgb_model.py              # XGBoostSurrogate
│   ├── rf_model.py               # RandomForestSurrogate
│   ├── gpr_model.py              # GPRSurrogate
│   └── __init__.py               # ALGORITHM_REGISTRY, ALGO_NAMES
│
├── modules/
│   ├── data_loading.py           # Tab 1: upload, preview, column selection
│   ├── preprocessing.py          # Tab 2: scaling, PCA, split, charts
│   ├── model_builder.py          # Tab 3: architecture config, training
│   ├── hyperopt.py               # Tab 4: Optuna HPO, visualization
│   ├── results.py                # Tab 5: metrics, SHAP, export
│   └── inference.py              # Tab 6: sensitivity analysis, batch predict
│
└── dataset/
    ├── dummy_dataset.csv
    ├── dummy_multi_output.xlsx
    ├── friedman1.xlsx / friedman2.xlsx / friedman3.xlsx
    └── ...
```

---

## License

MIT License — free to use, modify, and distribute.
