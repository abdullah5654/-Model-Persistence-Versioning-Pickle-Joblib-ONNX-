import os
import pickle
import joblib
import json
import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Optional ONNX
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ---------------------- CONFIG ----------------------
MODELS_DIR = "models"
CONFIG_FILE = "config.json"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------- TRAINING ----------------------
def train_models():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=200),
        "random_forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results[name] = {
            "model": model,
            "accuracy": acc
        }

    return results, X_test, y_test


# ---------------------- SAVE MODELS ----------------------
def save_model(model, name, version, method="pickle"):
    filename = f"{name}_v{version}"

    if method == "pickle":
        path = os.path.join(MODELS_DIR, f"{filename}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

    elif method == "joblib":
        path = os.path.join(MODELS_DIR, f"{filename}.joblib")
        joblib.dump(model, path)

    elif method == "onnx" and ONNX_AVAILABLE:
        path = os.path.join(MODELS_DIR, f"{filename}.onnx")
        initial_type = [("float_input", FloatTensorType([None, model.n_features_in_]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    else:
        return None

    return path


# ---------------------- VERSIONING ----------------------
def update_config(model_name, version, accuracy, path):
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

    config[f"{model_name}_v{version}"] = {
        "model_name": model_name,
        "version": version,
        "saved_path": path,
        "accuracy": accuracy,
        "date": str(datetime.datetime.now())
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    results, _, _ = train_models()

    version = 1
    for name, info in results.items():
        # Save with Pickle
        pickle_path = save_model(info["model"], name, version, method="pickle")
        update_config(name, version, info["accuracy"], pickle_path)

        # Save with Joblib
        joblib_path = save_model(info["model"], name, version, method="joblib")
        update_config(name, version, info["accuracy"], joblib_path)

        # Optional ONNX
        if ONNX_AVAILABLE:
            onnx_path = save_model(info["model"], name, version, method="onnx")
            update_config(name, version, info["accuracy"], onnx_path)

    print("✅ Training complete. Models and config saved!")
