# MLflow: Experiment Tracking, Model Registry, and Deployment

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, sharing and deploying models, and managing the model lifecycle in a central registry.

---

## Table of Contents

1. [Why MLflow?](#why-mlflow)
2. [Core Components](#core-components)
3. [Tracking Experiments](#tracking-experiments)
4. [Logging Metrics, Parameters, and Artifacts](#logging-metrics-parameters-and-artifacts)
5. [MLflow Projects](#mlflow-projects)
6. [MLflow Models and Flavors](#mlflow-models-and-flavors)
7. [Model Registry Lifecycle](#model-registry-lifecycle)
8. [MLflow Serve and Deployment](#mlflow-serve-and-deployment)
9. [Autologging](#autologging)
10. [Comparison: MLflow vs W&B vs Comet](#comparison-mlflow-vs-wb-vs-comet)
11. [Interview Q&A](#interview-qa)
12. [Common Pitfalls](#common-pitfalls)
13. [Related Topics](#related-topics)

---

## Why MLflow?

Machine learning development is inherently experimental. Teams run hundreds of experiments with different hyperparameters, datasets, and model architectures. Without a systematic tracking system:

- Results become impossible to reproduce
- It is unclear which model version is in production
- Knowledge is lost when team members leave
- Regulatory compliance becomes difficult

MLflow solves these problems by providing a unified platform that works with any ML library, language, or cloud.

**Key design principles:**
- Library-agnostic (works with scikit-learn, PyTorch, TensorFlow, XGBoost, etc.)
- Language-agnostic (Python, R, Java APIs)
- Cloud-agnostic (runs locally, on AWS, GCP, Azure, or Databricks)
- Open standard (uses open formats like YAML, JSON, MLmodel)

---

## Core Components

| Component | Purpose |
|---|---|
| MLflow Tracking | Log and query experiments (parameters, metrics, artifacts) |
| MLflow Projects | Package ML code in a reusable, reproducible format |
| MLflow Models | Package models for deployment across serving environments |
| MLflow Model Registry | Centralized model store with lifecycle management |

---

## Tracking Experiments

Every MLflow run belongs to an **experiment**. An experiment groups related runs together — for example, all runs tuning hyperparameters for a specific model.

```python
import mlflow

# Set the tracking URI (defaults to ./mlruns locally)
mlflow.set_tracking_uri("http://localhost:5000")  # Remote MLflow server
# mlflow.set_tracking_uri("sqlite:///mlflow.db")  # SQLite backend

# Create or set an experiment
mlflow.set_experiment("fraud-detection-v2")

# Start a run
with mlflow.start_run(run_name="xgboost-baseline") as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")

    # Your training code here
    ...
```

### Nested Runs

Useful for hyperparameter search where each trial is a child run:

```python
with mlflow.start_run(run_name="hyperparameter-search") as parent_run:
    for lr in [0.01, 0.001, 0.0001]:
        with mlflow.start_run(run_name=f"lr-{lr}", nested=True):
            mlflow.log_param("learning_rate", lr)
            # train and log metrics
```

### Querying Runs Programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs in an experiment
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.val_accuracy > 0.90",
    order_by=["metrics.val_accuracy DESC"],
    max_results=10
)

for run in runs:
    print(run.info.run_id, run.data.metrics["val_accuracy"])
```

---

## Logging Metrics, Parameters, and Artifacts

### Parameters

Parameters are key-value pairs that describe your run's configuration. They are logged once and are not expected to change during a run.

```python
with mlflow.start_run():
    # Log individual parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)

    # Log a dictionary of parameters at once
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8
    }
    mlflow.log_params(params)
```

### Metrics

Metrics are numeric values that can be updated over time (e.g., loss per epoch):

```python
import numpy as np

with mlflow.start_run():
    for epoch in range(100):
        train_loss = np.exp(-epoch * 0.05)
        val_loss = np.exp(-epoch * 0.04) + np.random.normal(0, 0.01)

        # Log with step for time-series visualization
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Log final summary metrics
    mlflow.log_metrics({
        "final_accuracy": 0.923,
        "final_f1": 0.918,
        "inference_latency_ms": 12.4
    })
```

### Artifacts

Artifacts are files associated with a run — model files, plots, datasets, reports:

```python
import matplotlib.pyplot as plt
import os

with mlflow.start_run():
    # Log a single file
    mlflow.log_artifact("model_config.yaml")

    # Log a directory
    mlflow.log_artifacts("outputs/", artifact_path="training_outputs")

    # Log a figure directly
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Training Curve")
    mlflow.log_figure(fig, "training_curve.png")

    # Log a dictionary as JSON
    mlflow.log_dict({"class_0": 0.95, "class_1": 0.89}, "per_class_metrics.json")

    # Log text
    mlflow.log_text("Model trained on 2024-01-15 dataset", "training_notes.txt")
```

### Tags

Tags are string key-value pairs used for metadata and filtering:

```python
with mlflow.start_run():
    mlflow.set_tag("team", "fraud-detection")
    mlflow.set_tag("data_version", "v2.3")
    mlflow.set_tag("mlflow.runName", "experiment-42")  # Built-in tag

    mlflow.set_tags({
        "environment": "staging",
        "git_commit": "a3f9b2c",
        "triggered_by": "nightly_job"
    })
```

---

## MLflow Projects

An MLflow Project is a convention for organizing ML code so it can be run reproducibly. It is defined by an `MLproject` file:

```yaml
# MLproject
name: fraud-detection

conda_env: conda.yaml
# OR: docker_env:
#       image: fraud-detection:latest

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 6}
      data_path: {type: str, default: "data/train.csv"}
    command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth} --data {data_path}"

  evaluate:
    parameters:
      model_uri: {type: str}
      test_data: {type: str}
    command: "python evaluate.py --model-uri {model_uri} --test-data {test_data}"
```

**Running a project:**

```bash
# Run locally
mlflow run . -P n_estimators=200 -P max_depth=8

# Run from GitHub
mlflow run https://github.com/org/fraud-detection -P n_estimators=200

# Run on Kubernetes
mlflow run . --backend kubernetes --backend-config kubernetes_config.json
```

---

## MLflow Models and Flavors

An MLflow Model is a directory containing the model files plus an `MLmodel` metadata file. A **flavor** is a model format that a particular tool understands.

### Built-in Flavors

| Flavor | Library |
|---|---|
| `mlflow.sklearn` | scikit-learn |
| `mlflow.pytorch` | PyTorch |
| `mlflow.tensorflow` | TensorFlow/Keras |
| `mlflow.xgboost` | XGBoost |
| `mlflow.lightgbm` | LightGBM |
| `mlflow.transformers` | Hugging Face Transformers |
| `mlflow.pyfunc` | Custom Python models |

### Saving and Loading Models

```python
from sklearn.ensemble import GradientBoostingClassifier
import mlflow.sklearn

# Training
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="FraudDetector",  # Automatically registers
        signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
        input_example=X_train[:5]  # Saves example for documentation
    )

# Loading for inference
model_uri = f"runs:/{run.info.run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Load as generic pyfunc (framework-agnostic)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
predictions = pyfunc_model.predict(X_test)
```

### Custom pyfunc Models

```python
import mlflow.pyfunc
import pandas as pd

class PreprocessAndPredict(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model_path"])
        self.scaler = joblib.load(context.artifacts["scaler_path"])

    def predict(self, context, model_input):
        scaled = self.scaler.transform(model_input)
        predictions = self.model.predict_proba(scaled)
        return pd.DataFrame(predictions, columns=["prob_legit", "prob_fraud"])

# Log the custom model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="custom_model",
        python_model=PreprocessAndPredict(),
        artifacts={
            "model_path": "model.joblib",
            "scaler_path": "scaler.joblib"
        },
        conda_env="conda.yaml"
    )
```

---

## Model Registry Lifecycle

The Model Registry provides a central hub for managing model versions and their lifecycle stages.

### Lifecycle Stages

```
None → Staging → Production → Archived
```

| Stage | Description |
|---|---|
| None | Newly registered, under evaluation |
| Staging | Candidate for production, undergoing testing |
| Production | Currently serving live traffic |
| Archived | Retired, kept for reference |

### Working with the Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model from a run
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="FraudDetector"
)
print(f"Version: {result.version}")

# Transition to staging
client.transition_model_version_stage(
    name="FraudDetector",
    version=result.version,
    stage="Staging",
    archive_existing_versions=False
)

# Add a description
client.update_model_version(
    name="FraudDetector",
    version=result.version,
    description="XGBoost model trained on Q4 2024 data. F1=0.918"
)

# Promote to production (archives existing production versions)
client.transition_model_version_stage(
    name="FraudDetector",
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

# Load the current production model
production_model = mlflow.pyfunc.load_model(
    model_uri="models:/FraudDetector/Production"
)

# Load a specific version
v3_model = mlflow.pyfunc.load_model(
    model_uri="models:/FraudDetector/3"
)
```

### Model Aliases (MLflow 2.x)

MLflow 2.x introduced aliases as a more flexible alternative to stages:

```python
# Set an alias
client.set_registered_model_alias("FraudDetector", "champion", version=5)
client.set_registered_model_alias("FraudDetector", "challenger", version=6)

# Load by alias
champion = mlflow.pyfunc.load_model("models:/FraudDetector@champion")
```

---

## MLflow Serve and Deployment

### Local Serving

```bash
# Serve a registered model
mlflow models serve -m "models:/FraudDetector/Production" --port 5001

# Serve from a run
mlflow models serve -m "runs:/abc123/model" --port 5001 --no-conda
```

**Making predictions via REST:**

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["feature1", "feature2"], "data": [[1.0, 2.0]]}}'
```

### Building Docker Images

```bash
# Build a Docker image for the model
mlflow models build-docker \
  -m "models:/FraudDetector/Production" \
  -n "fraud-detector:v1"

# Run the container
docker run -p 5001:8080 fraud-detector:v1
```

### Deployment to Cloud Platforms

```python
# Deploy to SageMaker
import mlflow.sagemaker as mfs

mfs.deploy(
    app_name="fraud-detector-prod",
    model_uri="models:/FraudDetector/Production",
    region_name="us-east-1",
    mode="create",  # or "replace", "add"
    execution_role_arn="arn:aws:iam::123456789:role/SageMakerRole",
    image_url="123456789.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest"
)
```

---

## Autologging

MLflow can automatically log parameters, metrics, and models without explicit log calls:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Enable autologging for all supported libraries
mlflow.autolog()

# OR enable for specific libraries
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    max_tuning_runs=5  # Limit nested runs for CV
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=6)
    model.fit(X_train, y_train)
    # MLflow automatically logs: n_estimators, max_depth, accuracy, F1, etc.
```

**Supported libraries for autologging:** scikit-learn, Keras, TensorFlow, PyTorch Lightning, XGBoost, LightGBM, Spark MLlib, Statsmodels, Fastai.

---

## Comparison: MLflow vs W&B vs Comet

| Feature | MLflow | Weights & Biases (W&B) | Comet ML |
|---|---|---|---|
| Open source | Yes (Apache 2.0) | No (SaaS, limited free tier) | No (SaaS, limited free tier) |
| Self-hosted | Yes (full control) | Yes (enterprise) | Yes (enterprise) |
| Experiment tracking | Yes | Yes (richer UI) | Yes |
| Model registry | Yes | Yes | Yes |
| Dataset versioning | Limited | Yes (Artifacts) | Yes |
| Hyperparameter sweep | Via Optuna/Ray | Built-in (Sweeps) | Built-in (Optimizer) |
| Collaboration UI | Basic | Excellent | Good |
| Deployment | Yes (serve, SageMaker) | Limited | Limited |
| Databricks integration | Native | Good | Good |
| Free tier | Fully free (self-hosted) | Limited runs | Limited compute |
| Learning curve | Medium | Low | Low |
| Best for | Enterprise, self-hosted, MLOps pipelines | Research teams, rich visualizations | Teams wanting SaaS with compliance |

**Recommendation:** Use MLflow when you need a fully open-source, self-hosted solution with strong deployment capabilities. Use W&B for research-heavy teams that value visualization. Use Comet when compliance and audit trails are critical.

---

## Interview Q&A

**Q1: What is the difference between an MLflow experiment and a run?**

An **experiment** is a logical grouping of related runs — for example, all attempts to build a fraud detection model. A **run** is a single execution of your training code with a specific set of parameters. One experiment contains many runs. You can think of an experiment as a folder and runs as files within it.

**Q2: How does the MLflow Model Registry differ from just storing model artifacts in a run?**

Run artifacts are just files stored in a blob store — there is no lifecycle management, versioning, or discoverability. The Model Registry adds: named versioning (v1, v2, v3), lifecycle stages (Staging/Production/Archived), descriptions, tags for governance, and aliases. It acts as a single source of truth for what model is in production, enabling CI/CD pipelines to fetch `models:/ModelName/Production` without hardcoding run IDs.

**Q3: What is an MLflow flavor and why does it matter?**

A flavor defines how a model can be loaded and used by a specific tool. For example, the `python_function` flavor allows any model to be loaded generically via `mlflow.pyfunc.load_model()`, while the `sklearn` flavor lets you load it with `mlflow.sklearn.load_model()` and get a native scikit-learn estimator. Flavors enable a single saved model to be consumed by multiple deployment targets (REST API, batch scoring, Spark UDF) without reformatting.

**Q4: How would you implement a blue-green deployment strategy using MLflow?**

Keep two registered models: `ModelName-Blue` (current production) and `ModelName-Green` (new version). After validating Green in staging, update your load balancer/feature flag to route traffic to the Green version. Only then archive Blue. Alternatively, use MLflow aliases: set `champion` to the current production version and `challenger` to the new one. Gradually shift traffic using your infrastructure layer, then update `champion` to point to the challenger version once validated.

**Q5: What is the `mlflow.models.infer_signature` function and why should you use it?**

`infer_signature` inspects sample input and output data to produce a typed schema (the model signature). This signature is stored in the `MLmodel` file and enables: input validation at serving time, automatic documentation generation, compatibility with MLflow's REST API data formats (splitting/record formats), and integration with model monitoring tools. Without a signature, the serving endpoint cannot validate incoming request shapes.

**Q6: How does MLflow handle experiment tracking in a distributed training environment?**

In distributed training (e.g., PyTorch DDP), only the rank-0 (master) process should call MLflow logging functions. In Horovod or DDP setups, wrap logging calls with `if hvd.rank() == 0:` or `if dist.get_rank() == 0:`. MLflow's tracking server is a centralized service accessed via HTTP, so all workers can technically log, but this causes duplicate entries. For Spark MLlib, MLflow's autologging handles this automatically.

**Q7: What backends does MLflow support for storing tracking data and artifacts?**

Tracking metadata (params, metrics, tags) can be stored in: local filesystem (`./mlruns`), SQLite, MySQL, PostgreSQL, or MSSQL. Artifacts (model files, plots) can be stored in: local filesystem, Amazon S3, Azure Blob Storage, Google Cloud Storage, SFTP, HDFS, or NFS. In production, you typically use a SQL database for metadata and cloud object storage for artifacts.

**Q8: How would you set up MLflow in a team environment?**

Deploy an MLflow Tracking Server with: a SQL database backend (PostgreSQL recommended) for metadata, cloud object storage (S3/GCS/Azure Blob) for artifacts, and optional authentication via a reverse proxy (nginx + OAuth). Use `mlflow.set_tracking_uri("http://mlflow-server:5000")` or the `MLFLOW_TRACKING_URI` environment variable. Set up the Model Registry on the same server. For access control, MLflow OSS has limited auth — consider Databricks MLflow or add an auth proxy.

**Q9: What is the difference between `mlflow.log_metric` with a `step` parameter and without?**

Without `step`, MLflow records the metric at `step=0` and overwrites it if called again with the same key. With `step`, MLflow stores a time-series of (step, value) pairs, enabling you to plot learning curves in the UI. Use `step=epoch` for training loops so you can see how loss or accuracy evolved over training.

**Q10: How do you reproduce an MLflow run exactly?**

An MLflow run stores: all parameters (`run.data.params`), the git commit hash (if logged), the conda/pip environment (if using MLflow Projects), and all artifacts including the source code (if logged). To reproduce: fetch the run via `client.get_run(run_id)`, check out the git commit, reconstruct the environment from the stored `conda.yaml` or `requirements.txt`, and re-run with the logged parameters. MLflow Projects make this fully automated via `mlflow run`.

**Q11: What are the limitations of MLflow autologging?**

Autologging can: produce excessive runs if not scoped properly (e.g., each cross-validation fold becomes a run), fail silently in some edge cases, log more than needed (increasing storage), not capture custom business metrics, and behave differently across library versions. It also does not log data lineage or dataset versions. Always combine autologging with explicit `log_metric` calls for business-critical metrics.

---

## Common Pitfalls

**1. Not setting a tracking URI in team environments**
Defaulting to `./mlruns` means each developer's runs are stored locally and invisible to teammates. Always set `MLFLOW_TRACKING_URI` in your shell profile or CI/CD environment.

**2. Logging inside a loop without steps**
Calling `mlflow.log_metric("loss", value)` inside a training loop without `step=epoch` overwrites the metric each time. You see only the final value and lose the training curve.

**3. Registering models without signatures or input examples**
Models without signatures cannot validate inputs at serving time. This leads to cryptic errors in production. Always use `infer_signature` or define the signature manually.

**4. Using stages instead of aliases for flexible routing**
The traditional stage system (Staging/Production) is rigid — only one model can be in Production at a time per model name. Aliases (MLflow 2.x) allow multiple simultaneously active versions with custom names, enabling champion/challenger setups.

**5. Storing large datasets as artifacts**
MLflow artifacts are not designed for versioning large datasets. Use a dedicated tool (DVC, Delta Lake, LakeFS) for dataset versioning and only log a reference (path or hash) in MLflow.

**6. Not archiving old production models**
Leaving multiple models in Production stage creates confusion about which is actually serving. Use `archive_existing_versions=True` when promoting a new version.

**7. Hardcoding run IDs in deployment scripts**
Scripts like `load_model("runs:/abc123/model")` break when a new model version is trained. Always load from the registry: `load_model("models:/ModelName/Production")`.

**8. Ignoring the conda environment in MLflow Projects**
Skipping the conda environment definition with `--no-conda` eliminates reproducibility. Always specify a `conda.yaml` or `python_env.yaml` for Projects used in production pipelines.

**9. Not tagging runs with metadata**
Without tags like `git_commit`, `data_version`, `triggered_by`, and `environment`, it becomes impossible to audit runs months later. Establish a tagging convention and enforce it in code templates.

**10. Running the tracking server without authentication**
The default MLflow server has no authentication. Anyone with network access can read and write experiments. Always place the server behind an authenticated reverse proxy or use a managed service.

---

## Related Topics

- [Intro to Kubeflow Pipelines](intro_kubeflow.md) — Orchestrating ML pipelines on Kubernetes
- [Intro to Model Serving](intro_model_serving.md) — TF Serving, Triton, BentoML, and serving strategies
- [Intro to Feature Stores](intro_feature_stores.md) — Feast, Tecton, and managing ML features
- [Intro to Model Explainability](intro_model_explainability.md) — SHAP, LIME, and interpretability
- [Intro to Data Quality](intro_data_quality.md) — Great Expectations, data contracts, and drift detection
- [Intro to CI/CD for ML](intro_cicd_ml.md) — Automating ML pipelines with GitHub Actions and Jenkins
