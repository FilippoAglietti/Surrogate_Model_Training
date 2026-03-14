import numpy as np
from modules.model_builder import build_surrogate_model, get_keras_loss, get_keras_optimizer

# --- 1. Single-output model ---
model = build_surrogate_model(
    input_dim=5, output_dim=1, num_layers=2, neurons=32,
    act_hidden="ReLU", act_out="Linear",
    dropout_rate=0.1, l1_val=0.0, l2_val=0.0
)
criterion = get_keras_loss("MeanSquaredError")
optimizer = get_keras_optimizer("Adam", 0.001)
model.compile(optimizer=optimizer, loss=criterion)
print("Single-output model built and compiled.")

X = np.random.rand(20, 5).astype(np.float32)
y = np.random.rand(20, 1).astype(np.float32)
history = model.fit(X, y, epochs=3, verbose=0)
preds = model.predict(X, verbose=0)
assert preds.shape == (20, 1), f"Expected (20,1), got {preds.shape}"
print(f"Single-output fit OK. Final loss: {history.history['loss'][-1]:.6f}")

# --- 2. Multi-output model ---
model_multi = build_surrogate_model(
    input_dim=5, output_dim=3, num_layers=2, neurons=32,
    act_hidden="ReLU", act_out="Linear",
    dropout_rate=0.0, l1_val=0.0, l2_val=0.0
)
model_multi.compile(optimizer=get_keras_optimizer("Adam", 0.001), loss=get_keras_loss("MeanSquaredError"))
y_multi = np.random.rand(20, 3).astype(np.float32)
model_multi.fit(X, y_multi, epochs=3, verbose=0)
preds_multi = model_multi.predict(X, verbose=0)
assert preds_multi.shape == (20, 3), f"Expected (20,3), got {preds_multi.shape}"
print(f"Multi-output fit OK. Shape: {preds_multi.shape}")

# --- 3. Regularization + dropout ---
model_reg = build_surrogate_model(
    input_dim=5, output_dim=1, num_layers=3, neurons=64,
    act_hidden="LeakyReLU", act_out="Linear",
    dropout_rate=0.2, l1_val=1e-4, l2_val=1e-4
)
model_reg.compile(optimizer=get_keras_optimizer("RMSprop", 0.001), loss=get_keras_loss("Huber"))
model_reg.fit(X, y, epochs=3, verbose=0)
print("Regularization + dropout model OK.")

print("\nALL SMOKE TESTS PASSED.")
