# ⚡ Surrogate Model Training Engine

A modular ML training & optimization tool for building **surrogate models** using Neural Networks. Built with **Streamlit**, **PyTorch**, and **Optuna** — featuring a dark neon terminal-style GUI.

![Python](https://img.shields.io/badge/Python-3.9+-00ff41?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-00e5ff?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff6e40?style=flat-square&logo=streamlit&logoColor=white)

---

## 🚀 Quick Start

### Windows
```
run.bat
```
The batch file will:
1. Create a Python virtual environment (if not found)
2. Install all dependencies
3. Launch the Streamlit app at `http://localhost:8501`

### Manual Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
streamlit run app.py
```

---

## 📋 Features

| Module | Description |
|--------|-------------|
| **📂 Data Loading** | Upload Excel/CSV, interactive column selection, data preview |
| **🔧 Preprocessing** | NaN handling, Min-Max/Standard normalization, train/val/test split |
| **🏗 Model Builder** | Dynamic NN architecture (layers, activations, dropout), loss & optimizer config |
| **🔍 Hyperopt** | Optuna (TPE), Random Search, Grid Search with live progress |
| **🚀 Training** | Real-time loss curves, R² metrics, early stopping, epoch log |
| **📊 Results** | Predicted vs Actual, residuals, feature importance, export CSV/Excel/model |

---

## 🛠 Tech Stack

- **Frontend**: Streamlit (dark neon theme, JetBrains Mono, ASCII art)
- **ML Backend**: PyTorch (feedforward neural networks)
- **Optimization**: Optuna (TPE sampler), scikit-learn
- **Visualization**: Plotly (interactive dark charts)
- **Data**: Pandas, NumPy, openpyxl

---

## 📁 Project Structure

```
Surrogate_Model_Training/
├── app.py                  # Main entry point
├── run.bat                 # Windows launcher (auto venv)
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Dark theme config
├── utils/
│   ├── theme.py            # Neon CSS + ASCII art
│   └── state.py            # Session state helpers
└── modules/
    ├── data_loading.py     # File upload & column selection
    ├── preprocessing.py    # NaN, normalization, splitting
    ├── model_builder.py    # NN architecture builder
    ├── hyperopt.py         # HPO (Optuna/Random/Grid)
    ├── training.py         # Training loop + dashboard
    └── results.py          # Plots, metrics, exports
```

---

## 📖 Usage

1. **Data Loading** — Upload your `.xlsx` or `.csv` file, select input features and target output
2. **Preprocessing** — Handle missing values, normalize data, configure train/val/test split ratios
3. **Model Builder** — Design your neural network architecture layer-by-layer
4. *(Optional)* **Hyperopt** — Run automated hyperparameter search
5. **Training** — Train the model with real-time loss visualization
6. **Results** — Analyze predictions, download results and trained model

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
