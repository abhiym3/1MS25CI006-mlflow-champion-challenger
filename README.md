# Champion–Challenger Sentiment Comparison with Hugging Face and MLflow

## Task and models
- **Task:** Binary Sentiment Classification (`sst2`, first 50 validation rows, `seed=42`)
- **Model A (Current Champion):** `distilbert-base-uncased-finetuned-sst-2-english`
- **Model B (Challenger):** `textattack/bert-base-uncased-SST-2` (mapped `LABEL_0 -> NEGATIVE`, `LABEL_1 -> POSITIVE`)

## How to run
1. Create a virtual environment and install dependencies:
   `pip install -r requirements.txt`
2. Start MLflow tracking server:
   `python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000`
3. Run evaluation and logging:
   `python evaluate.py --usn 1MS25CI006 --samples 50 --seed 42`
4. Register models and set aliases:
   `python register_champion.py`
5. Run predictions on new data using the `@champion` alias:
   `python predict_from_mlflow.py`

## Champion rule (written before the final run)
- **Primary metric:** `macro_f1` (higher is better; secondary metrics: `accuracy` higher is better, `avg_latency_ms` and `total_inference_s` lower is better).
- **Minimum improvement margin:** Challenger (`B`) must beat Champion (`A`) by at least **+0.02** (`2.0` percentage points in `macro_f1`) to replace `A`. On a small 50-sample subset, smaller gaps can easily come from borderline examples or noise.
- **Tie-breaker / Guardrail:** If `|delta_macro_f1| < 0.02`, keep the current Champion (`A`). Additionally, reject any Challenger whose `avg_latency_ms` exceeds **5x** the Champion's `avg_latency_ms`.

## Results table
| Model | Role | Macro-F1 (higher) | Accuracy (higher) | Avg Latency ms (lower) | Total Time s (lower) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `distilbert-base-uncased-finetuned-sst-2-english` | Champion candidate (A) | TODO | TODO | TODO | TODO |
| `textattack/bert-base-uncased-SST-2` | Challenger (B) | TODO | TODO | TODO | TODO |
| **Delta (B - A)** | — | TODO | TODO | TODO | TODO |

## Decision
TODO: Write 150–250 words after running `evaluate.py` and `register_champion.py` covering whether B improved on A, by how much, whether the difference is meaningful given the sample size, and trade-offs in speed, model size, and error types.

## Limitations and declaration
- Small evaluation subset (`50` examples) on CPU; results may vary on larger or domain-shifted reviews.
- Built with assistance from Gemini; all code and results were reviewed and tested locally.

## Results table
| Model | Role | Macro-F1 (higher) | Accuracy (higher) | Avg Latency ms (lower) | Total Time s (lower) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `distilbert-base-uncased-finetuned-sst-2-english` | Champion candidate (A) | 0.8572 | 0.8600 | 9.47 | 0.47 |
| `textattack/bert-base-uncased-SST-2` | Challenger (B) | 0.9167 | 0.9200 | 16.74 | 0.84 |
| **Delta (B - A)** | — | +0.0595 | +0.0600 | +7.27 | +0.36 |

## Decision
Under our pre-declared rule, Challenger Model B (`textattack/bert-base-uncased-SST-2`) is promoted to `@champion` and Model A (`distilbert-base-uncased-finetuned-sst-2-english`) becomes `@challenger`. Model B improves `macro_f1` from `0.8572` to `0.9167` (`+0.0595`, or roughly +5.95 percentage points) and accuracy from `0.8600` to `0.9200` (`+0.0600`) on the 50-sample SST-2 validation subset. Because `+0.0595` clears our `+0.02` minimum improvement margin and B's average latency (`16.74 ms` vs. `9.47 ms`, about `1.77x` A's latency) stays well below the `5x` guardrail, B wins cleanly. At the same time, 50 samples is a relatively small evaluation set, so part of the gain could shift on a larger or noisier dataset; we also should note that `textattack/bert-base-uncased-SST-2` is larger (~110M vs. ~67M parameters), runs slower per item, and outputs `LABEL_0`/`LABEL_1` labels that require explicit mapping to `NEGATIVE`/`POSITIVE` and careful handling of borderline or context-heavy reviews.