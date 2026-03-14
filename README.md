# ⚡ Surrogate Model Training Engine V3

A modular ML training & optimization tool for building **surrogate models** using Neural Networks. Built natively with **CustomTkinter**, **TensorFlow/Keras**, and **Optuna** — featuring a dark neon aesthetic.

![Python](https://img.shields.io/badge/Python-3.9+-00ff41?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-00e5ff?style=flat-square&logo=tensorflow&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2+-ff6e40?style=flat-square&logo=python&logoColor=white)

---

## 🚀 Quick Start

### Windows
```
run.bat
```
The batch file will:
1. Create a Python virtual environment (if not found)
2. Install all dependencies
3. Launch the desktop app using `python app.py`

### Manual Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
python app.py
```

---

## 📋 Features

| Module | Description |
|--------|-------------|
| **📂 Data Loading** | Upload Excel/CSV, **Scrollable Preview Table**, **Missing Values bar chart**, **Column type badges** (NUM/CAT/DATE), Multi-Output selection |
| **🔧 Preprocessing** | Train/Val/Test split, MinMax/Standard Scaling, PCA for Inputs/Outputs, **Scree Plot**, **Combined Matrix** (scatter+corr+hist), **Box/Violin**, **KDE**, **Parallel Coordinates**, **Outlier Detection & Removal**, **💾 Save Image** on every chart |
| **🏗 Model Builder** | Dynamic Multi-Output NN architecture, loss & optimizer config, **LR Scheduler** (CosineDecay / ExponentialDecay), **Reduce LR on Plateau**, **Model Summary panel**, live training dash, **💾 Save training curve** |
| **🔍 Hyperopt** | Threaded **Optuna** (TPE) automated search, **Best Trials Table**, **Param Importances**, **Parallel Coordinates**, **Contour Plot**, **💾 Save Image** on every chart |
| **📊 Results** | Pred vs Actual, Test Index Series, Residuals, **Q-Q Plot**, **Per-Target Metrics table**, **Worst Predictions panel**, **SHAP feature importance**, Deployable Model Wrapper (.zip), **💾 Save Image** on every chart |
| **🔮 Inference** | **1D Sensitivity** (sliders + curves + base-point marker), **2D Sensitivity Contour Grid** (select 1 output + up to 5 inputs → C(5,2) contour plots), Batch Excel prediction, **💾 Save Image** on every chart |

---

## 📁 Project Structure

```
Surrogate_Model_Training/
├── app.py                      # Main CustomTkinter Application
├── run.bat                     # Windows auto-venv launcher
├── requirements.txt            # Dependencies
├── smoke_test.py               # Integration test (single-output, multi-output, regularization)
├── generate_dataset.py         # Generates Friedman benchmark datasets (single-output)
├── generate_multi_dataset.py   # Generates a test multi-output dataset
├── utils/
│   ├── theme.py                # Global styling constants
│   ├── state.py                # Singleton application state dictionary
│   └── plot_utils.py           # Shared "Save Image" button helper
└── modules/
    ├── data_loading.py
    ├── preprocessing.py
    ├── model_builder.py
    ├── hyperopt.py
    ├── results.py
    └── inference.py
```

## 📄 License
MIT License
