import os
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-champion-challenger"
NEW_DATA_PATH = os.path.join("data", "new_data.csv")
OUT_PATH = os.path.join("outputs", "new_data_predictions.csv")

DEFAULT_NEW_EXAMPLES = [
    "This lab setup actually worked on the first try and saved me hours.",
    "The server crashed right before demo day and it was super frustrating.",
    "Nothing special here, just an average weekend movie.",
    "Clean documentation and fast inference made testing a breeze.",
    "Too slow on CPU and gave completely wrong labels on short phrases.",
    "Great team support during the hackathon sprint.",
]


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(NEW_DATA_PATH):
        pd.DataFrame({"text": DEFAULT_NEW_EXAMPLES}).to_csv(NEW_DATA_PATH, index=False)

    new_df = pd.read_csv(NEW_DATA_PATH)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    alias_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    model_version_obj = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    source_run = client.get_run(model_version_obj.run_id)
    hf_model_id = source_run.data.params.get("hf_model_id", "unknown")

    loaded_model = mlflow.pyfunc.load_model(alias_uri)
    preds_df = loaded_model.predict(new_df[["text"]])
    preds_df.to_csv(OUT_PATH, index=False)

    print(f"Loaded URI: {alias_uri}")
    print(f"Registered Version: v{model_version_obj.version} | Run ID: {model_version_obj.run_id}")
    print(f"Hugging Face Model ID: {hf_model_id}\n")
    print(preds_df.to_string(index=False))
    print(f"\nSaved predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()