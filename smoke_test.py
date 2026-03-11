import numpy as np
from modules.model_builder import build_surrogate_model
from modules.hyperopt import get_keras_loss, get_keras_optimizer

# 1. Test model building
layers = [{"units": 32, "activation": "ReLU", "dropout": 0.1}]
model = build_surrogate_model(5, 1, layers)
print("Model built natively in Keras:", type(model))

# 2. Test compiling
criterion = get_keras_loss("MeanSquaredError")
optimizer = get_keras_optimizer("Adam", 0.001)
model.compile(optimizer=optimizer, loss=criterion)
print("Model compiled gracefully.")

# 3. Test dummy pass
X_dummy = np.random.rand(10, 5).astype(np.float32)
y_dummy = np.random.rand(10, 1).astype(np.float32)

history = model.fit(X_dummy, y_dummy, epochs=2, verbose=0)
print("Model fit successful. Final loss:", history.history['loss'][-1])

# 4. Test predicting
preds = model.predict(X_dummy, verbose=0)
print("Predictions generated. Shape:", preds.shape)

print("ALL SMOKE TESTS PASSED.")
