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
| **Free tier** | SageMaker Studio (limited) | $300 credit for new users | $200 credit for new users |

---

## Feature Comparison

| Feature | SageMaker | Vertex AI | Azure ML |
|---------|----------|----------|---------|
| **Managed notebooks** | Studio Notebooks | Workbench | Compute Instances / Studio |
| **Training jobs** | Training Jobs (built-in containers) | Custom Training | Compute Clusters |
| **AutoML** | Autopilot | AutoML (Tables, Vision, NLP) | Automated ML |
| **Model registry** | Model Registry | Model Registry | Model Registry |
| **Feature store** | SageMaker Feature Store | Vertex AI Feature Store | Azure ML Feature Store |
| **Pipelines / MLflow** | SageMaker Pipelines | Vertex AI Pipelines (Kubeflow) | Azure ML Pipelines |
| **Real-time endpoints** | Real-time Endpoints | Online Prediction | Real-time Endpoints |
| **Batch inference** | Batch Transform | Batch Prediction | Batch Endpoints |
| **Model monitoring** | Model Monitor | Skew/Drift Detection | Data drift monitoring |
| **Experiment tracking** | Experiments (basic) / MLflow | Vertex AI Experiments | MLflow integrated |
| **A/B testing** | Multi-variant endpoints | Traffic split endpoints | Mirroring / A/B |
| **Edge deployment** | SageMaker Edge Manager | Google Edge TPU | Azure IoT Edge |
| **Distributed training** | Built-in (Horovod, FSDP) | Reduction Server | Distributed training |
| **Spot/preemptible** | Managed Spot Training | Preemptible VMs | Low-priority compute |

---

## AWS SageMaker

SageMaker is the most comprehensive ML platform with the most managed components.

### Key Components

```
SageMaker Studio          → Web-based IDE (notebooks, experiments, pipelines)
SageMaker Training Jobs   → Managed training with built-in algorithms or custom containers
SageMaker Endpoints       → Real-time inference with auto-scaling
SageMaker Pipelines       → ML workflow orchestration (CI/CD for ML)
SageMaker Model Registry  → Version and approve models before deployment
SageMaker Feature Store   → Offline (S3) + Online (low-latency) feature serving
SageMaker Model Monitor   → Data quality, model quality, bias, explainability monitoring
SageMaker Clarify         → Bias detection and explainability
SageMaker Autopilot       → AutoML: automatic model selection and tuning
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
    use_spot_instances=True,       # Up to 90% cost savings
    max_wait=3600,
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
from sagemaker.workflow.parameters import ParameterInteger, ParameterString

# Pipeline parameters
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

# Define steps
step_train = TrainingStep(name="TrainModel", estimator=estimator, inputs={...})
step_register = RegisterModel(name="RegisterModel", estimator=estimator,
                               model_approval_status=model_approval_status)

# Create pipeline
pipeline = Pipeline(
    name="FraudDetectionPipeline",
    parameters=[model_approval_status],
    steps=[step_train, step_register],
)
pipeline.upsert(role_arn=role)
pipeline.start()
```

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
Vertex AI AutoML          → No-code training for tabular, vision, NLP, video
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
from kfp import dsl
from kfp.v2 import compiler
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
    joblib.dump(model, model_output.path + '/model.pkl')

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
Real-time Endpoints       → Managed Kubernetes-based serving
Batch Endpoints           → Large-scale batch inference
Automated ML              → AutoML for tabular, vision, NLP
Azure ML Designer         → Drag-and-drop ML pipeline builder
MLflow Integration        → Built-in MLflow for experiment tracking
Responsible AI Dashboard  → Fairness, explainability, error analysis
Azure OpenAI Service      → OpenAI models (GPT-4, DALL-E) in Azure
```

### Training with MLflow Tracking

```python
import mlflow
import mlflow.sklearn
from azureml.core import Workspace, Experiment
from sklearn.ensemble import RandomForestClassifier

# Connect to Azure ML workspace
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name="fraud-detection")

with mlflow.start_run() as run:
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", 100)

    # Register model
    mlflow.sklearn.log_model(model, "fraud_model", registered_model_name="fraud-detector")

# Deploy to real-time endpoint
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice

model = Model(ws, 'fraud-detector', version=1)
service = Model.deploy(ws, "fraud-endpoint", [model],
                       deployment_config=AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1))
service.wait_for_deployment(show_output=True)
```

### Automated ML

```python
from azureml.train.automl import AutoMLConfig
from azureml.core import Experiment

automl_config = AutoMLConfig(
    task='classification',
    primary_metric='AUC_weighted',
    training_data=train_dataset,
    label_column_name='is_fraud',
    n_cross_validations=5,
    iterations=50,
    iteration_timeout_minutes=5,
    enable_early_stopping=True,
    featurization='auto',
    enable_onnx_compatible_models=True,
)

experiment = Experiment(ws, "automl-fraud")
run = experiment.submit(automl_config)
run.wait_for_completion(show_output=True)

best_run, fitted_model = run.get_output()
print(f"Best model: {best_run.get_properties()['algorithm']}")
```

---

## When to Choose Each Platform

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Existing AWS infrastructure | **SageMaker** | Native S3, IAM, VPC integration |
| BigQuery as data warehouse | **Vertex AI** | Direct BigQuery connector, no data movement |
| Azure Active Directory / compliance | **Azure ML** | Enterprise integration, compliance certifications |
| Best AutoML for tabular data | **Vertex AI AutoML** | Generally best accuracy and features |
| LLM fine-tuning / GenAI | **Vertex AI** or **SageMaker** | Model Garden vs JumpStart |
| Responsible AI & fairness | **Azure ML** | Best responsible AI dashboard |
| Cheapest for small experiments | **Vertex AI** | Generous free tier for notebooks |
| Most built-in algorithms | **SageMaker** | 17+ built-in algorithms |
| Kubeflow-based pipelines | **Vertex AI Pipelines** | Native Kubeflow support |

---

## Cost Optimization Strategies

| Strategy | SageMaker | Vertex AI | Azure ML |
|----------|----------|----------|---------|
| Spot/preemptible instances | Managed Spot Training (up to 90% off) | Preemptible VMs (up to 80% off) | Low-priority compute (up to 80% off) |
| Auto-scaling to zero | Serverless Inference | Min replicas = 0 | Scale-down enabled clusters |
| Right-sizing instances | Instance Recommender | Recommender | Compute SKU comparison |
| Multi-model endpoints | Multi-Model Endpoints | Model Registry | Multiple models per endpoint |
| Caching predictions | ElastiCache integration | Cloud Memorystore | Azure Cache for Redis |

```python
# SageMaker: Managed Spot Training
estimator = SKLearn(
    ...,
    use_spot_instances=True,      # Use spot instances
    max_run=3600,                  # Max training time in seconds
    max_wait=7200,                 # Max wait including spot interruptions
)

# Azure ML: Low-priority compute cluster
from azureml.core.compute import AmlCompute, ComputeTarget

compute_config = AmlCompute.provisioning_configuration(
    vm_size='STANDARD_D3_V2',
    vm_priority='lowpriority',     # Low-priority = significant cost savings
    min_nodes=0,                   # Scale to zero when idle
    max_nodes=4,
)
```

---

## Interview Q&A

**Q1: What are the key differences between SageMaker and Vertex AI?**
SageMaker has greater breadth and more managed services (17+ built-in algorithms, dedicated endpoints, SageMaker Clarify for bias). Vertex AI has tighter BigQuery integration (no data movement for tabular data), generally better AutoML accuracy, and is built on open standards (Kubeflow Pipelines, TFX). SageMaker is better if you're already on AWS; Vertex AI if you're on GCP with BigQuery as your warehouse.

**Q2: What is the purpose of a model registry in a cloud ML platform?**
A model registry centralizes model versioning and lifecycle management. It stores trained model artifacts, metadata (metrics, hyperparameters, training data version), approval status (pending/approved/rejected), and deployment history. It enables governance: only approved models go to production, and you can trace which model version is currently serving.

**Q3: How would you design a CI/CD pipeline for ML on AWS SageMaker?**
1. Code commit triggers GitHub Actions
2. Build and push Docker training image to ECR
3. Run SageMaker Training Job
4. Evaluate model — if metrics pass threshold, register in Model Registry with "PendingApproval"
5. Manual approval step (or automated if metrics exceed threshold)
6. Upon approval, SageMaker Pipelines deploys to staging endpoint
7. Integration tests on staging
8. Promote to production endpoint with canary deployment

**Q4: What is the difference between online prediction and batch prediction endpoints?**
Online (real-time) endpoints are always-running services that respond to individual requests within milliseconds — used for interactive applications. Batch prediction endpoints process large datasets efficiently (millions of records in parallel) on a schedule — used for pre-computing predictions (daily scoring runs). Online: higher cost (always on), low latency. Batch: cost-efficient, high latency acceptable.

**Q5: How do AutoML platforms differ from custom model training?**
AutoML automatically searches the model architecture and hyperparameter space — no ML expertise required, faster time-to-first-model. Custom training: full control over architecture, features, and optimization — better ceiling performance but requires more expertise. AutoML is best for establishing a baseline, quick prototyping, and non-ML teams. Custom training is best when you need maximum accuracy or have unique domain requirements.

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Not using spot/preemptible instances | 3-5x higher training costs | Default to spot for training; save on-demand for serving |
| Endpoints running 24/7 at full capacity | Expensive waste during low traffic | Enable auto-scaling with min=0 for dev/staging |
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
| [Kubernetes](../devops/intro_kubernetes.md) | Azure ML and Vertex AI use K8s for endpoints |
| [Docker](../devops/intro_docker.md) | Custom containers are the basis for cloud ML training |
| [Study Pattern](../docs/study-pattern.md) | Cloud ML Platforms are an Advanced (🔴) topic |
