import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# --- Step 1: Train a Native Scikit-Learn Model ---
print("1. Training Native Scikit-Learn Model...")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32) # ONNX usually expects float32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("   Model trained successfully.")


# --- Step 2: Export Model to ONNX Format ---
print("\n2. Exporting to ONNX...")

# Define the input type (4 features for Iris, float32)
initial_type = [('float_input', FloatTensorType([None, 4]))]

# Convert the scikit-learn model to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model to a file
onnx_file_path = "iris_rf.onnx"
with open(onnx_file_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"   Model saved to {onnx_file_path}")


# --- Step 3: Inference with ONNX Runtime ---
print("\n3. Setting up ONNX Runtime...")

# Create an inference session
session = ort.InferenceSession(onnx_file_path)

# Get input name (needed for the session run method)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name

print(f"   Input Name: {input_name}")
print(f"   Output Name: {label_name}")


# --- Step 4: Benchmarking Performance ---
print("\n4. Benchmarking: Native vs. ONNX Runtime...")

# We will run inference on the test set multiple times to measure average speed
iterations = 10000

# Benchmark Native Scikit-Learn
start_time = time.time()
for _ in range(iterations):
    model.predict(X_test)
native_duration = time.time() - start_time

# Benchmark ONNX Runtime
# Note: ONNX Runtime expects inputs as a dictionary
start_time = time.time()
for _ in range(iterations):
    session.run([label_name], {input_name: X_test})
onnx_duration = time.time() - start_time

# --- Results ---
print("-" * 30)
print(f"Results over {iterations} iterations on test set:")
print(f"Native Scikit-Learn Time: {native_duration:.4f} seconds")
print(f"ONNX Runtime Time:      {onnx_duration:.4f} seconds")

speedup = native_duration / onnx_duration
print(f"Speedup Factor:         {speedup:.2f}x")
print("-" * 30)
