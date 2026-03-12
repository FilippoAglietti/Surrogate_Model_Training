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
| **📂 Data Loading** | Upload Excel/CSV, Multi-Output Selection, basic data stats |
| **🔧 Preprocessing** | Train/Val/Test split, MinMax/Standard Scaling, **PCA for Inputs/Outputs**, Correlation Heatmaps |
| **🏗 Model Builder** | Dynamic Multi-Output NN architecture, loss & optimizer config, live training dash |
| **🔍 Hyperopt** | Threaded **Optuna** (TPE) automated search with Matplotlib history plots |
| **📊 Results** | Dynamic Pred vs Actual grid plots, Test Index Series, Residuals, **Deployable Model Wrapper (.zip)** export |
| **🔮 Inference** | **Interactive Sensitivity Analysis** (sliders + dynamic curves) & Batch Excel prediction |

---

## 📁 Project Structure

```
Surrogate_Model_Training/
├── app.py                  # Main CustomTkinter Application
├── run.bat                 # Windows auto-venv launcher
├── requirements.txt        # Dependencies
├── generate_multi_dataset.py # Script to create a test multi-output dataset
├── utils/
│   ├── theme.py            # Global styling constants
│   └── state.py            # Singleton application state dictionary
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
