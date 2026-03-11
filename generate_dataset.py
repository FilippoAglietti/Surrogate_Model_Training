"""
Generate Friedman synthetic datasets for testing the Surrogate Model Trainer.
Uses sklearn.datasets.make_friedman1/2/3.
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3

os.makedirs("dataset", exist_ok=True)

# ── Friedman #1 ──────────────────────────────────────────────
# y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise
X, y = make_friedman1(n_samples=500, n_features=5, noise=0.5, random_state=42)
df1 = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(5)])
df1["y"] = y
df1.to_excel("dataset/friedman1.xlsx", index=False)
print(f"✓ friedman1.xlsx  — {df1.shape[0]} rows, {df1.shape[1]} cols")

# ── Friedman #2 ──────────────────────────────────────────────
# y = sqrt(x1^2 + (x2*x3 - 1/(x2*x4))^2) + noise
X, y = make_friedman2(n_samples=500, noise=0.5, random_state=42)
df2 = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
df2["y"] = y
df2.to_excel("dataset/friedman2.xlsx", index=False)
print(f"✓ friedman2.xlsx  — {df2.shape[0]} rows, {df2.shape[1]} cols")

# ── Friedman #3 ──────────────────────────────────────────────
# y = atan((x2*x3 - 1/(x2*x4)) / x1) + noise
X, y = make_friedman3(n_samples=500, noise=0.5, random_state=42)
df3 = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
df3["y"] = y
df3.to_excel("dataset/friedman3.xlsx", index=False)
print(f"✓ friedman3.xlsx  — {df3.shape[0]} rows, {df3.shape[1]} cols")

print("\nAll datasets saved to ./dataset/")
