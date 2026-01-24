from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi import HTTPException
import joblib
import json
import tensorflow as tf
import os


# ---------------------------
# Create FastAPI app
# ---------------------------
app = FastAPI(
    title="Water Quality Prediction API",
    description="FastAPI backend for water quality & infrastructure anomaly prediction",
    version="1.0"
)

class EngineeredInput(BaseModel):
    features: Dict[str, float]

# ---------------------------
# Load models & artifacts
# ---------------------------
MODEL_DIR = "models"

# Preprocessor
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))

# Feature columns
with open(os.path.join(MODEL_DIR, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

# Feature group indices
feature_group_indices = joblib.load(
    os.path.join(MODEL_DIR, "feature_group_indices.joblib")
)

time_idx  = feature_group_indices["time"]
phys_idx  = feature_group_indices["phys"]
inter_idx = feature_group_indices["inter"]
stats_idx = feature_group_indices["stats"]
flags_idx = feature_group_indices["flags"]

# XGBoost models
xgb_reg = joblib.load(os.path.join(MODEL_DIR, "xgb_regression.joblib"))
xgb_clf = joblib.load(os.path.join(MODEL_DIR, "xgb_classification.joblib"))
xgb_anom = joblib.load(os.path.join(MODEL_DIR, "xgb_anomaly.joblib"))

# Isolation Forest
iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))

# Label encoder
wqi_encoder = joblib.load(os.path.join(MODEL_DIR, "wqi_label_encoder.joblib"))

# Neural network
nn_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "multi_task_model.keras"),
    compile=False
)

print("✅ All models loaded successfully")


def predict_from_engineered_df(engineered_df):
    """
    Takes engineered DataFrame and returns predictions
    """


    import pandas as pd

    # Ensure all required features exist
    for col in feature_columns:
        if col not in engineered_df.columns:
            engineered_df[col] = 0.0  # safe default

    # Reorder columns exactly as training
    X = engineered_df[feature_columns]


    # Preprocess
    X_proc = preprocessor.transform(X)

    results = {}

    # ------------------
    # Model 1: XGB Regression
    # ------------------
    results["CCME_Value_XGB"] = float(
        xgb_reg.predict(X_proc)[0]
    )

    # ------------------
    # Model 2: XGB Classification
    # ------------------
    cls_probs = xgb_clf.predict_proba(X_proc)[0]
    cls_idx = cls_probs.argmax()

    results["CCME_WQI_Class"] = wqi_encoder.inverse_transform([cls_idx])[0]
    results["CCME_WQI_Probabilities"] = cls_probs.tolist()

    # ------------------
    # Model 3: Anomaly (XGB + Isolation Forest)
    # ------------------
    results["Infra_Anomaly_XGB"] = float(
        xgb_anom.predict_proba(X_proc)[0][1]
    )

    results["Infra_Anomaly_IForest"] = int(
        iso_forest.predict(X_proc)[0] == -1
    )

    # ------------------
    # Model 4: Neural Network
    # ------------------
    X_time  = X_proc[:, time_idx]
    X_phys  = X_proc[:, phys_idx]
    X_inter = X_proc[:, inter_idx]
    X_stats = X_proc[:, stats_idx]
    X_flags = X_proc[:, flags_idx]

    preds = nn_model(
        [X_time, X_phys, X_inter, X_stats, X_flags],
        training=False
    )

    results["CCME_Value_NN"] = float(preds[0].numpy()[0][0])
    results["CCME_WQI_NN_Prob"] = preds[1].numpy()[0].tolist()
    results["Infra_Anomaly_NN"] = float(preds[2].numpy()[0][0])

    return results


# ---------------------------
# Test endpoint
# ---------------------------
@app.get("/")
def root():
    return {"message": "API is running and models are loaded"}


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "models_loaded": {
            "xgb_regression": True,
            "xgb_classification": True,
            "xgb_anomaly": True,
            "isolation_forest": True,
            "neural_network": True
        },
        "api_version": "1.0"
    }



@app.post("/predict")
def predict(data: EngineeredInput):
    """
    Receives engineered features and returns predictions
    """
    import pandas as pd

    try:
        input_df = pd.DataFrame([data.features])
        results = predict_from_engineered_df(input_df)
        return results

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
