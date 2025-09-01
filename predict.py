import argparse
import json
import pickle
import joblib
import numpy as np

CONFIG_FILE = "config.json"

def load_model(version_key):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    if version_key not in config:
        raise ValueError(f"Version {version_key} not found in config.json")

    path = config[version_key]["saved_path"]

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
    elif path.endswith(".joblib"):
        model = joblib.load(path)
    else:
        raise ValueError("Unsupported format (only Pickle/Joblib supported here)")

    return model

def main():
    parser = argparse.ArgumentParser(description="Predict with saved model")
    parser.add_argument("--version", type=str, required=True, help="Model version key (e.g., logistic_regression_v1)")
    parser.add_argument("--input", type=str, required=True, help="Comma-separated feature values (e.g., '5.1,3.5,1.4,0.2')")
    args = parser.parse_args()

    model = load_model(args.version)
    sample = np.array([list(map(float, args.input.split(",")))]).reshape(1, -1)
    prediction = model.predict(sample)

    print(f"✅ Prediction: {prediction[0]}")

if __name__ == "__main__":
    main()
