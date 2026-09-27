# Cloud ML Platforms Comparison

AWS SageMaker, Google Vertex AI, and Azure Machine Learning are the three major managed ML platforms. Choosing the right platform depends on your existing cloud ecosystem, team expertise, and specific feature requirements.

---

## Table of Contents
1. [Platform Overview](#platform-overview)
2. [Feature Comparison](#feature-comparison)
3. [AWS SageMaker](#aws-sagemaker)
4. [Google Vertex AI](#google-vertex-ai)
5. [Azure Machine Learning](#azure-machine-learning)
6. [When to Choose Each Platform](#when-to-choose-each-platform)
7. [Cost Optimization Strategies](#cost-optimization-strategies)
8. [Interview Q&A](#interview-qa)
9. [Common Pitfalls](#common-pitfalls)
10. [Related Topics](#related-topics)

---

## Platform Overview

| | AWS SageMaker | Google Vertex AI | Azure ML |
|-|--------------|-----------------|----------|
| **Cloud** | AWS | GCP | Azure |
| **Launched** | 2017 | 2021 (unified) | 2018 |
| **Strength** | Breadth, scale, enterprise | AutoML, built-in data tools | Azure integration, responsible AI |
| **LLM/GenAI** | Bedrock + JumpStart | Vertex AI Studio + Gemini | Azure OpenAI Service |
| **Pricing model** | Pay-per-use | Pay-per-use | Pay-per-use |
| **Free tier** | Limited free tier for new accounts | Trial credits for new accounts | Trial credits for new accounts |

---

## Feature Comparison

| Feature | SageMaker | Vertex AI | Azure ML |
|---------|----------|----------|---------|
| **Managed notebooks** | Studio (JupyterLab / Code Editor spaces) | Workbench instances | Compute Instances / Studio |
| **Training jobs** | Training Jobs (built-in containers) | Custom Training | Compute Clusters |
| **AutoML** | Autopilot (now surfaced in SageMaker Canvas) | AutoML (tabular, image, video) | Automated ML |
| **Model registry** | Model Registry | Model Registry | Model Registry |
| **Feature store** | SageMaker Feature Store | Vertex AI Feature Store | Azure ML Feature Store |
| **Pipelines / MLflow** | SageMaker Pipelines | Vertex AI Pipelines (Kubeflow) | Azure ML Pipelines |
| **Real-time endpoints** | Real-time Endpoints | Online Prediction | Managed Online Endpoints |
| **Batch inference** | Batch Transform | Batch Prediction | Batch Endpoints |
| **Model monitoring** | Model Monitor | Model Monitoring (skew/drift) | Model monitoring (drift, data quality) |
| **Experiment tracking** | Experiments (basic) / MLflow | Vertex AI Experiments | MLflow integrated |
| **A/B testing** | Production variants | Traffic split endpoints | Traffic split / mirroring |
| **Edge deployment** | Edge Manager was discontinued in 2024 (use AWS IoT Greengrass) | AutoML edge model export / Coral Edge TPU | Azure IoT Edge |
| **Distributed training** | Distributed training libraries, PyTorch FSDP | Reduction Server | Distributed training |
| **Spot/preemptible** | Managed Spot Training | Spot VMs (successor to preemptible) | Low-priority / spot VMs |

---

## AWS SageMaker

SageMaker is the broadest ML platform, with the most managed components. Since late 2024 AWS brands the ML service as **Amazon SageMaker AI**; "Amazon SageMaker" now also covers SageMaker Unified Studio for data, analytics, and AI.

### Key Components

```
SageMaker Studio          → Web-based IDE (notebooks, experiments, pipelines; replaced Studio Classic)
SageMaker Training Jobs   → Managed training with built-in algorithms or custom containers
SageMaker Endpoints       → Real-time inference with auto-scaling
SageMaker Pipelines       → ML workflow orchestration (CI/CD for ML)
SageMaker Model Registry  → Version and approve models before deployment
SageMaker Feature Store   → Offline (S3) + Online (low-latency) feature serving
SageMaker Model Monitor   → Data quality, model quality, bias, explainability monitoring
SageMaker Clarify         → Bias detection and explainability
SageMaker Autopilot       → AutoML: automatic model selection and tuning (now part of SageMaker Canvas)
SageMaker JumpStart       → Pre-trained models (foundation models, fine-tuning)
```

### Training Example

```python
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import get_execution_role

role = get_execution_role()
session = sagemaker.Session()

# Define the estimator
estimator = SKLearn(
    entry_point='train.py',        # Your training script
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    framework_version='1.2-1',
    py_version='py3',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 5,
    },
    use_spot_instances=True,       # Large savings vs on-demand (check current pricing)
    max_run=3600,
    max_wait=7200,                 # Must be >= max_run
)

# Train
estimator.fit({'train': 's3://my-bucket/train/', 'test': 's3://my-bucket/test/'})

# Deploy
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
)

# Predict
result = predictor.predict([[25, 50000, 1, 0]])
print(result)
```

### SageMaker Pipelines

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterInteger, ParameterString

# Pipeline parameters
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

# Define steps
step_train = TrainingStep(name="TrainModel", estimator=estimator, inputs={...})
step_register = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    model_package_group_name="FraudDetectionModels",
    approval_status=model_approval_status,
)

# Create pipeline
pipeline = Pipeline(
    name="FraudDetectionPipeline",
    parameters=[model_approval_status],
    steps=[step_train, step_register],
)
pipeline.upsert(role_arn=role)
pipeline.start()
```

Newer SageMaker Python SDK releases favor `ModelStep` with `step_args` over `RegisterModel`, and add `ModelTrainer` / `ModelBuilder` interfaces; check the SDK version you pin.

---

## Google Vertex AI

Vertex AI is Google's unified ML platform, tight integration with BigQuery and Google's AI research.

### Key Components

```
Vertex AI Workbench       → Managed JupyterLab notebooks
Vertex AI Training        → Custom training on GCP infrastructure
Vertex AI Prediction      → Online (real-time) and Batch endpoints
Vertex AI Pipelines       → Kubeflow Pipelines-based ML orchestration
Vertex AI Feature Store   → Managed feature store with BigQuery backend
Vertex AI Model Registry  → Register, version, and manage models
Vertex AI Experiments     → Track runs, metrics, parameters
Vertex AI AutoML          → No-code training for tabular, image, video (text use cases moved to Gemini tuning)
Vertex AI Studio          → Prompt design and LLM experimentation (Gemini)
Vertex AI Search          → Enterprise search and recommendations
Model Garden              → Pre-trained models (Gemini, Llama, etc.)
```

### Training Example

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Custom training job
job = aiplatform.CustomTrainingJob(
    display_name="fraud-detection-training",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-2:latest",
    requirements=["scikit-learn==1.2.0", "pandas==2.0.0"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
)

model = job.run(
    dataset=dataset,
    model_display_name="fraud-detector",
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",  # Optional GPU
    accelerator_count=1,
    replica_count=1,
    args=["--n_estimators=100", "--max_depth=5"],
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=5,  # Auto-scaling
)

# Predict
prediction = endpoint.predict(instances=[[25, 50000, 1, 0]])
```

### Vertex AI Pipelines

```python
from kfp import compiler, dsl
from google.cloud.aiplatform import pipeline_jobs

@dsl.component(packages_to_install=['scikit-learn', 'pandas'])
def train_model(data_path: str, model_output: dsl.Output[dsl.Model]):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = pd.read_csv(data_path)
    X, y = df.drop('label', axis=1), df['label']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, model_output.path)  # artifact path is a file path

@dsl.pipeline(name="fraud-detection-pipeline")
def fraud_pipeline(data_path: str = "gs://my-bucket/data.csv"):
    train_task = train_model(data_path=data_path)

compiler.Compiler().compile(fraud_pipeline, "pipeline.json")

pipeline_job = pipeline_jobs.PipelineJob(
    display_name="fraud-detection",
    template_path="pipeline.json",
    pipeline_root="gs://my-bucket/pipeline-root",
)
pipeline_job.run()
```

---

## Azure Machine Learning

Azure ML integrates tightly with the Azure ecosystem (Azure DevOps, Azure Data Factory, Synapse Analytics) and has strong responsible AI features.

### Key Components

```
Azure ML Studio           → Web-based UI for all ML tasks
Compute Instances         → Managed notebook VMs
Compute Clusters          → Scalable training clusters (auto-scale to 0)
Azure ML Pipelines        → ML workflow orchestration
Model Registry            → Version and deploy models
Online Endpoints          → Managed online endpoints (or Kubernetes online endpoints on your own AKS / Arc cluster)
Batch Endpoints           → Large-scale batch inference
Automated ML              → AutoML for tabular, vision, NLP
Azure ML Designer         → Drag-and-drop pipeline builder (current designer uses custom components; classic prebuilt components are legacy v1)
MLflow Integration        → Built-in MLflow for experiment tracking
Responsible AI Dashboard  → Fairness, explainability, error analysis
Azure OpenAI              → OpenAI models hosted in Azure, managed through Azure AI Foundry (now Microsoft Foundry)
```

### Training with MLflow Tracking (SDK v2)

The v1 SDK (`azureml-core`, `Workspace`, `Experiment`, `AciWebservice`) is deprecated and past its announced end of support, and ACI deployment from Azure ML is legacy. Use SDK v2 (`azure-ai-ml`, `MLClient`) and managed online endpoints.

```python
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential
from sklearn.ensemble import RandomForestClassifier

# Connect to Azure ML workspace
ml_client = MLClient.from_config(credential=DefaultAzureCredential())
ws = ml_client.workspaces.get(ml_client.workspace_name)
mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
mlflow.set_experiment("fraud-detection")

with mlflow.start_run():
    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", 100)

    # Register model
    mlflow.sklearn.log_model(model, "fraud_model", registered_model_name="fraud-detector")

# Deploy to a managed online endpoint (MLflow models need no scoring script)
endpoint = ManagedOnlineEndpoint(name="fraud-endpoint", auth_mode="key")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="fraud-endpoint",
    model="azureml:fraud-detector:1",
    instance_type="Standard_DS3_v2",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

### Automated ML (SDK v2)

```python
from azure.ai.ml import Input, automl

classification_job = automl.classification(
    compute="cpu-cluster",
    experiment_name="automl-fraud",
    training_data=Input(type="mltable", path="azureml:fraud-train:1"),
    target_column_name="is_fraud",
    primary_metric="AUC_weighted",
    n_cross_validations=5,
)
classification_job.set_limits(
    max_trials=50,
    trial_timeout_minutes=5,
    enable_early_termination=True,
)
classification_job.set_featurization(mode="auto")
classification_job.set_training(enable_onnx_compatible_models=True)

returned_job = ml_client.jobs.create_or_update(classification_job)
ml_client.jobs.stream(returned_job.name)
# Inspect the best trial in Studio or via MLflow on the parent run
```

---

## When to Choose Each Platform

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Existing AWS infrastructure | **SageMaker** | Native S3, IAM, VPC integration |
| BigQuery as data warehouse | **Vertex AI** | Direct BigQuery connector, no data movement |
| Microsoft Entra ID (formerly Azure AD) / compliance | **Azure ML** | Enterprise integration, compliance certifications |
| Strong managed AutoML for tabular data | **Vertex AI AutoML** or **Azure Automated ML** | Mature tabular AutoML; benchmark on your own data |
| LLM fine-tuning / GenAI | **Vertex AI** or **SageMaker** | Model Garden vs JumpStart |
| Responsible AI & fairness | **Azure ML** | Built-in Responsible AI dashboard |
| Cheapest for small experiments | **Depends** | Free tiers and trial credits change; idle notebooks usually dominate cost |
| Most built-in algorithms | **SageMaker** | Large catalog of built-in algorithms |
| Kubeflow-based pipelines | **Vertex AI Pipelines** | Native Kubeflow support |

---

## Cost Optimization Strategies

| Strategy | SageMaker | Vertex AI | Azure ML |
|----------|----------|----------|---------|
| Spot/preemptible instances | Managed Spot Training | Spot VMs | Low-priority VMs |
| Auto-scaling to zero | Serverless Inference, or scale to zero where supported | Batch prediction, or scale to zero where supported | Clusters with min nodes = 0 |
| Right-sizing instances | Inference Recommender | Recommender | Compute SKU comparison |
| Multi-model endpoints | Multi-Model Endpoints | Shared deployment resource pools | Multiple deployments per endpoint |
| Caching predictions | ElastiCache integration | Cloud Memorystore | Azure Cache for Redis |

```python
# SageMaker: Managed Spot Training
estimator = SKLearn(
    ...,
    use_spot_instances=True,      # Use spot instances
    max_run=3600,                  # Max training time in seconds
    max_wait=7200,                 # Max wait including spot interruptions
)

# Azure ML (SDK v2): Low-priority compute cluster
from azure.ai.ml.entities import AmlCompute

cluster = AmlCompute(
    name='cpu-cluster',
    size='Standard_D3_v2',
    tier='low_priority',           # Low-priority = significant cost savings
    min_instances=0,               # Scale to zero when idle
    max_instances=4,
    idle_time_before_scale_down=300,
)
ml_client.compute.begin_create_or_update(cluster).result()
```

---

## Interview Q&A

**Q1: What are the key differences between SageMaker and Vertex AI?**
SageMaker has greater breadth and more managed services (many built-in algorithms, several endpoint types, SageMaker Clarify for bias). Vertex AI has tighter BigQuery integration (less data movement for tabular data), strong AutoML, and is built on open standards (Kubeflow Pipelines, TFX). SageMaker is better if you're already on AWS; Vertex AI if you're on GCP with BigQuery as your warehouse.

**Q2: What is the purpose of a model registry in a cloud ML platform?**
A model registry centralizes model versioning and lifecycle management. It stores trained model artifacts, metadata (metrics, hyperparameters, training data version), approval status (pending/approved/rejected), and deployment history. It enables governance: only approved models go to production, and you can trace which model version is currently serving.

**Q3: How would you design a CI/CD pipeline for ML on AWS SageMaker?**
1. Code commit triggers GitHub Actions
2. Build and push Docker training image to ECR
3. Run SageMaker Training Job
4. Evaluate model: if metrics pass threshold, register in Model Registry with "PendingManualApproval"
5. Manual approval step (or automated if metrics exceed threshold)
6. Upon approval, an EventBridge rule on the approval status change triggers deployment (e.g. via CodePipeline) to a staging endpoint
7. Integration tests on staging
8. Promote to production endpoint with canary deployment

**Q4: What is the difference between online prediction and batch prediction endpoints?**
Online (real-time) endpoints are always-running services that respond to individual requests within milliseconds: used for interactive applications. Batch prediction endpoints process large datasets efficiently (millions of records in parallel) on a schedule: used for pre-computing predictions (daily scoring runs). Online: higher cost (always on), low latency. Batch: cost-efficient, high latency acceptable.

**Q5: How do AutoML platforms differ from custom model training?**
AutoML automatically searches the model architecture and hyperparameter space: no ML expertise required, faster time-to-first-model. Custom training: full control over architecture, features, and optimization: better ceiling performance but requires more expertise. AutoML is best for establishing a baseline, quick prototyping, and non-ML teams. Custom training is best when you need maximum accuracy or have unique domain requirements.

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Not using spot/preemptible instances | Much higher training costs | Default to spot for training; save on-demand for serving |
| Endpoints running 24/7 at full capacity | Expensive waste during low traffic | Enable auto-scaling; scale to zero or use serverless/batch for dev/staging where supported |
| Storing data on instance storage | Data lost on shutdown | Use S3, GCS, or Azure Blob for all datasets |
| No pipeline versioning | Can't reproduce training runs | Use SageMaker/Vertex pipelines with parameter versioning |
| Choosing platform before cloud commitment | Vendor lock-in without benefit | Align with existing cloud infrastructure investment |
| Not monitoring endpoint drift | Silent model degradation | Enable Model Monitor / Skew Detection from day one |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [MLflow](../mlops/intro_mlflow.md) | All three platforms support MLflow for experiment tracking |
| [Model Serving](../mlops/intro_model_serving.md) | Endpoints are the cloud platform's serving layer |
| [Feature Stores](../mlops/intro_feature_stores.md) | All three have managed feature stores |
| [Kubernetes](../devops/intro_kubernetes.md) | Managed endpoints hide the cluster, but Azure ML can also deploy to your own Kubernetes (AKS / Arc) |
| [Docker](../devops/intro_docker.md) | Custom containers are the basis for cloud ML training |
| [Study Pattern](../docs/study-pattern.md) | Cloud ML Platforms are an Advanced (🔴) topic |
