import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-champion-challenger"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run evaluate.py first.")

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=20)
    run_a = next((r for r in runs if r.info.run_name.startswith("A_")), None)
    run_b = next((r for r in runs if r.info.run_name.startswith("B_")), None)
    run_cmp = next((r for r in runs if r.info.run_name.startswith("compare_")), None)

    if not run_a or not run_b or not run_cmp:
        raise RuntimeError("Could not find A_, B_, or compare_ runs.")

    mv_a = mlflow.register_model(model_uri=f"runs:/{run_a.info.run_id}/model", name=REGISTERED_MODEL_NAME)
    mv_b = mlflow.register_model(model_uri=f"runs:/{run_b.info.run_id}/model", name=REGISTERED_MODEL_NAME)

    winner = run_cmp.data.tags.get("winner", "champion_A")
    if winner == "challenger_B":
        champ_ver, chall_ver = mv_b.version, mv_a.version
    else:
        champ_ver, chall_ver = mv_a.version, mv_b.version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", champ_ver)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", chall_ver)

    print(f"Registered '{REGISTERED_MODEL_NAME}': @champion -> v{champ_ver}, @challenger -> v{chall_ver}")


if __name__ == "__main__":
    main()