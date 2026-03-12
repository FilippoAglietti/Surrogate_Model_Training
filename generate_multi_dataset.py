import pandas as pd
import numpy as np
import os

def generate():
    np.random.seed(42)
    n_samples = 1500
    
    # Inputs
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(0, 10, n_samples)
    x3 = np.random.normal(50, 15, n_samples)
    x4 = np.random.choice([0, 1, 2, 3], n_samples)
    x5 = np.random.uniform(-1, 1, n_samples)

    # Outputs
    # Y1: Non-linear combination
    y1 = 3 * np.sin(x1) + (x2 ** 2) / 10 - x5 * 5 + np.random.normal(0, 0.5, n_samples)
    
    # Y2: Linear with some interaction
    y2 = 2 * x1 - 0.5 * x3 + 5 * x4 + np.random.normal(0, 1.0, n_samples)
    
    # Y3: Complex interaction
    y3 = np.exp(x2 / 5) * x5 + np.cos(x3 / 10) * 10 + np.random.normal(0, 0.2, n_samples)

    df = pd.DataFrame({
        "Input_X1_Angle": x1,
        "Input_X2_Speed": x2,
        "Input_X3_Temp": x3,
        "Input_X4_Mode": x4,
        "Input_X5_Friction": x5,
        "Target_Y1_Efficiency": y1,
        "Target_Y2_Wear": y2,
        "Target_Y3_Noise": y3,
    })

    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dummy_multi_output.xlsx")
    df.to_excel(out_path, index=False)
    print(f"Generated multi-output dataset at: {out_path}")

if __name__ == "__main__":
    generate()
