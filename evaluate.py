import argparse
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.pyfunc
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
import torch
from transformers import pipeline

warnings.filterwarnings("ignore")

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "POSITIVE": "POSITIVE",
}


class SentimentPyFuncWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id: str, batch_size: int = 16, device: int = -1):
        self.hf_model_id = hf_model_id
        self.batch_size = batch_size
        self.device = device
        self.pipe = None

    def load_context(self, context):
        self.pipe = pipeline(
            task="sentiment-analysis",
            model=self.hf_model_id,
            tokenizer=self.hf_model_id,
            device=self.device,
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        texts = model_input["text"].astype(str).tolist()
        raw_preds = self.pipe(texts, batch_size=self.batch_size, truncation=True)
        labels = [LABEL_MAP.get(item["label"], item["label"]) for item in raw_preds]
        scores = [float(item["score"]) for item in raw_preds]
        return pd.DataFrame({"text": texts, "prediction": labels, "score": scores})


def run_model_eval(hf_model_id: str, texts: list[str], refs: list[str], batch_size: int, device: int):
    pipe = pipeline(
        task="sentiment-analysis",
        model=hf_model_id,
        tokenizer=hf_model_id,
        device=device,
    )
    _ = pipe(texts[:2], batch_size=batch_size, truncation=True)

    t0 = time.perf_counter()
    raw_preds = pipe(texts, batch_size=batch_size, truncation=True)
    total_s = time.perf_counter() - t0
    avg_ms = (total_s / len(texts)) * 1000.0

    preds = [LABEL_MAP.get(item["label"], item["label"]) for item in raw_preds]
    scores = [float(item["score"]) for item in raw_preds]

    acc = float(accuracy_score(refs, preds))
    macro_f1 = float(f1_score(refs, preds, average="macro", zero_division=0))

    return preds, scores, {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "avg_latency_ms": float(avg_ms),
        "total_inference_s": float(total_s),
    }


def make_plots(refs, preds_a, preds_b, metrics_a, metrics_b, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["NEGATIVE", "POSITIVE"]
    for ax, preds, title in zip(axes, [preds_a, preds_b], ["Model A (Champion)", "Model B (Challenger)"]):
        cm = confusion_matrix(refs, preds, labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    metric_names = ["macro_f1", "accuracy"]
    vals_a = [metrics_a[m] for m in metric_names]
    vals_b = [metrics_b[m] for m in metric_names]
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, vals_a, width, label="Model A")
    ax.bar(x + width / 2, vals_b, width, label="Model B")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Champion vs Challenger Quality Metrics")
    ax.legend()
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "metrics_comparison.png")
    plt.savefig(bar_path, dpi=150)
    plt.close(fig)

    return cm_path, bar_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usn", default="1MS25CI006")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    ds = load_dataset("nyu-mll/glue", "sst2", split="validation").shuffle(seed=args.seed).select(range(args.samples))
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    texts = [str(x) for x in ds["sentence"]]
    refs = [id2label[int(y)] for y in ds["label"]]

    eval_df = pd.DataFrame({"id": range(len(texts)), "text": texts, "reference": refs})
    eval_path = os.path.join("data", "eval_inputs.csv")
    eval_df.to_csv(eval_path, index=False)

    model_a_id = "distilbert-base-uncased-finetuned-sst-2-english"
    model_b_id = "textattack/bert-base-uncased-SST-2"
    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"

    preds_a, scores_a, metrics_a = run_model_eval(model_a_id, texts, refs, args.batch_size, device)
    preds_b, scores_b, metrics_b = run_model_eval(model_b_id, texts, refs, args.batch_size, device)

    df_a = eval_df.copy()
    df_a["prediction"] = preds_a
    df_a["score"] = scores_a
    df_a["correct"] = df_a["prediction"] == df_a["reference"]
    pred_a_path = os.path.join("outputs", "predictions_model_a.csv")
    df_a.to_csv(pred_a_path, index=False)

    df_b = eval_df.copy()
    df_b["prediction"] = preds_b
    df_b["score"] = scores_b
    df_b["correct"] = df_b["prediction"] == df_b["reference"]
    pred_b_path = os.path.join("outputs", "predictions_model_b.csv")
    df_b.to_csv(pred_b_path, index=False)

    disagreements = []
    for idx in range(len(texts)):
        if preds_a[idx] != preds_b[idx]:
            disagreements.append({
                "id": idx,
                "text": texts[idx],
                "reference": refs[idx],
                "model_a_pred": preds_a[idx],
                "model_b_pred": preds_b[idx],
                "comment": "Models predicted opposite sentiment on this review.",
            })
    disagree_path = os.path.join("outputs", "disagreements.csv")
    pd.DataFrame(disagreements[:5]).to_csv(disagree_path, index=False)

    metrics_summary = {"model_a": metrics_a, "model_b": metrics_b}
    metrics_path = os.path.join("outputs", "metrics_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    cm_path, bar_path = make_plots(refs, preds_a, preds_b, metrics_a, metrics_b, "outputs")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    common_tags = {
        "student_usn": args.usn,
        "library_versions": f"transformers={__import__('transformers').__version__},torch={torch.__version__},mlflow={mlflow.__version__}",
        "dataset_name": "glue_sst2_validation_subset",
    }

    sample_input = pd.DataFrame({"text": texts[:3]})
    sample_output = pd.DataFrame({
        "text": texts[:3],
        "prediction": preds_a[:3],
        "score": scores_a[:3],
    })
    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run(run_name="A_sentiment_distilbert-sst2") as run_a:
        mlflow.log_params({
            "hf_model_id": model_a_id,
            "task": "sentiment-classification",
            "role": "champion_candidate",
            "batch_size": args.batch_size,
            "eval_samples": args.samples,
            "seed": args.seed,
            "device": device_name,
        })
        mlflow.set_tags(common_tags)
        mlflow.log_metrics(metrics_a)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=SentimentPyFuncWrapper(model_a_id, args.batch_size, device),
            signature=signature,
            input_example=sample_input,
        )
        mlflow.log_artifact(pred_a_path)
        mlflow.log_artifact(eval_path)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(bar_path)

    with mlflow.start_run(run_name="B_sentiment_bert-base-sst2") as run_b:
        mlflow.log_params({
            "hf_model_id": model_b_id,
            "task": "sentiment-classification",
            "role": "challenger_candidate",
            "batch_size": args.batch_size,
            "eval_samples": args.samples,
            "seed": args.seed,
            "device": device_name,
        })
        mlflow.set_tags(common_tags)
        mlflow.log_metrics(metrics_b)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=SentimentPyFuncWrapper(model_b_id, args.batch_size, device),
            signature=signature,
            input_example=sample_input,
        )
        mlflow.log_artifact(pred_b_path)
        mlflow.log_artifact(eval_path)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(bar_path)

    margin = 0.02
    delta_f1 = metrics_b["macro_f1"] - metrics_a["macro_f1"]
    latency_ratio = metrics_b["avg_latency_ms"] / max(metrics_a["avg_latency_ms"], 1e-6)
    challenger_wins = (delta_f1 >= margin) and (latency_ratio <= 5.0)
    winner_label = "challenger_B" if challenger_wins else "champion_A"

    with mlflow.start_run(run_name="compare_sentiment") as run_cmp:
        mlflow.log_params({
            "primary_metric": "macro_f1",
            "margin": margin,
            "max_latency_ratio_guardrail": 5.0,
            "run_a_id": run_a.info.run_id,
            "run_b_id": run_b.info.run_id,
        })
        mlflow.set_tags({**common_tags, "winner": winner_label})
        mlflow.log_metric("delta_macro_f1", float(delta_f1))
        mlflow.log_metric("latency_ratio_b_to_a", float(latency_ratio))
        mlflow.log_artifact(disagree_path)
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")

    print(f"Done. Winner per rule: {winner_label} | delta_macro_f1={delta_f1:.4f} | run_cmp={run_cmp.info.run_id}")


if __name__ == "__main__":
    main()