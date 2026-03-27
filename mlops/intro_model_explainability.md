# Model Explainability: SHAP, LIME, and Interpretability Techniques

Model explainability (also called interpretability or XAI — Explainable AI) refers to the degree to which humans can understand the causes of a model's decisions. As ML models are deployed in high-stakes domains — credit scoring, medical diagnosis, hiring, criminal justice — the ability to explain predictions has become both a technical and regulatory necessity.

---

## Table of Contents

1. [Why Explainability Matters](#why-explainability-matters)
2. [Taxonomy of Explanations](#taxonomy-of-explanations)
3. [Feature Importance (Permutation and Impurity-Based)](#feature-importance)
4. [SHAP Values](#shap-values)
5. [SHAP Plots](#shap-plots)
6. [LIME](#lime)
7. [Integrated Gradients](#integrated-gradients)
8. [Partial Dependence Plots and ICE Curves](#partial-dependence-plots-and-ice-curves)
9. [Model Cards](#model-cards)
10. [Regulatory Compliance and the EU AI Act](#regulatory-compliance-and-the-eu-ai-act)
11. [Interview Q&A](#interview-qa)
12. [Common Pitfalls](#common-pitfalls)
13. [Related Topics](#related-topics)

---

## Why Explainability Matters

**Trust and adoption:** Stakeholders — doctors, loan officers, judges — will not act on a model's output unless they can understand and verify its reasoning.

**Debugging:** Explainability reveals data leakage, spurious correlations, and feature engineering errors faster than aggregate metrics. A model predicting hospital readmission based on "discharge day = Friday" reveals a systematic bias that accuracy scores would miss.

**Fairness and bias auditing:** Understanding which features drive predictions allows identification of proxy discrimination (e.g., zip code as a proxy for race in lending).

**Regulatory compliance:** GDPR Article 22 grants individuals the right to an explanation for automated decisions. The EU AI Act mandates transparency for high-risk AI systems. US financial regulations (ECOA, FCRA) require adverse action notices explaining credit denials.

**Model improvement:** Feature importance explanations guide feature engineering and data collection priorities.

---

## Taxonomy of Explanations

| Dimension | Options |
|---|---|
| Scope | Global (entire model) vs. Local (single prediction) |
| Model dependency | Model-agnostic vs. Model-specific |
| Output type | Feature importance, rule extraction, example-based, counterfactual |
| Fidelity | Exact (inherent) vs. Approximate (post-hoc surrogate) |

**Global explanations** describe overall model behavior: "Which features does this model rely on most across the training set?"

**Local explanations** describe a single prediction: "Why did the model reject this specific loan application?"

---

## Feature Importance

### Impurity-Based (Mean Decrease in Impurity)

Built into tree-based models. Each feature's importance is the total reduction in node impurity (Gini or entropy) weighted by the number of samples that pass through that node.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

importances.head(15).plot(kind="bar", figsize=(12, 5))
plt.title("Impurity-Based Feature Importances")
plt.tight_layout()
plt.show()
```

**Limitation:** Biased toward high-cardinality features (many unique values). A random ID column can appear important.

### Permutation Importance

More reliable: measures how much model performance drops when a feature's values are randomly shuffled, breaking its relationship with the target.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val,
    n_repeats=30,
    random_state=42,
    scoring="roc_auc"
)

perm_df = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

print(perm_df.head(10))
```

**Limitation:** Correlated features split importance between themselves, underestimating the combined importance of a feature group.

---

## SHAP Values

SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory. For a prediction, the Shapley value of a feature is its **average marginal contribution** across all possible subsets (coalitions) of features.

**Key property (SHAP axioms):**
- **Efficiency:** All SHAP values sum to the difference between the prediction and the base rate (expected model output)
- **Symmetry:** Features with identical contributions get identical SHAP values
- **Dummy:** Features that never affect any prediction get SHAP value zero
- **Linearity:** SHAP values are additive across models

### The Core Formula

```
f(x) = base_value + SHAP_1 + SHAP_2 + ... + SHAP_n
```

Where `base_value = E[f(X)]` is the average prediction across the training set.

### TreeSHAP

TreeSHAP is an exact, polynomial-time algorithm for tree-based models (decision trees, random forests, XGBoost, LightGBM, CatBoost). It runs in O(TLD²) time where T is the number of trees, L is maximum leaves, and D is maximum depth.

```python
import shap
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Create the SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)
# For classifiers: shap_values may be a list [class_0_shap, class_1_shap]
# or a single array (XGBoost log-odds output)

print(f"SHAP values shape: {shap_values.shape}")  # (n_samples, n_features)
print(f"Base value: {explainer.expected_value:.4f}")

# Verify additivity for one sample
sample_idx = 0
prediction = model.predict_proba(X_test[sample_idx:sample_idx+1])[0, 1]
shap_sum = explainer.expected_value + shap_values[sample_idx].sum()
print(f"Model prediction: {prediction:.4f}")
print(f"Base + SHAP sum: {shap_sum:.4f}")  # Should match (in log-odds space for XGB)
```

### KernelSHAP

Model-agnostic approximation using weighted linear regression on feature coalitions. Much slower than TreeSHAP — use only when TreeSHAP is not available.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True))])
pipeline.fit(X_train, y_train)

# Use a background dataset (summarized for speed)
background = shap.kmeans(X_train, 50)  # 50 representative samples

explainer = shap.KernelExplainer(pipeline.predict_proba, background)

# Compute SHAP values for a small test set (KernelSHAP is slow)
shap_values = explainer.shap_values(X_test[:100], nsamples=500)
```

### DeepSHAP

For neural networks — uses a modified backpropagation approach based on DeepLIFT:

```python
import torch
import shap

# background = sample of training data as tensor
background = torch.tensor(X_train[:100], dtype=torch.float32)
test_input = torch.tensor(X_test[:10], dtype=torch.float32)

explainer = shap.DeepExplainer(pytorch_model, background)
shap_values = explainer.shap_values(test_input)
```

---

## SHAP Plots

### Summary Plot (Global)

Shows feature importance and direction of effect across the whole dataset:

```python
import shap

# Bar plot: mean absolute SHAP value per feature
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Beeswarm plot: each point is one sample, color = feature value
shap.summary_plot(shap_values, X_test)
# Red = high feature value, Blue = low feature value
# X-axis = SHAP value (impact on model output)
```

**Reading a beeswarm plot:** A feature with red dots on the right means high values of that feature increase the prediction. Blue dots on the left means low values of that feature decrease the prediction.

### Waterfall Plot (Local)

Explains a single prediction by showing each feature's contribution from the base value to the final prediction:

```python
# Explain a single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=X_test.columns.tolist()
))
```

### Force Plot (Local)

An alternative single-prediction visualization that shows forces pushing toward and away from the base value:

```python
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    matplotlib=True
)
```

### Dependence Plot

Shows how one feature's SHAP value changes as the feature value changes, with interaction effects colored by a second feature:

```python
# Automatic interaction feature selection
shap.dependence_plot("age", shap_values, X_test)

# Explicitly set interaction feature
shap.dependence_plot("income", shap_values, X_test, interaction_index="credit_score")
```

---

## LIME

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the black-box model locally with an interpretable model (typically a sparse linear model).

**Algorithm:**
1. Take the instance to explain
2. Generate perturbed samples around it
3. Get the black-box model's predictions for those samples
4. Weight samples by proximity to the original instance
5. Fit a sparse linear model on the weighted samples
6. Return coefficients as feature importances for that prediction

```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Legitimate", "Fraud"],
    mode="classification",
    discretize_continuous=True
)

# Explain a single prediction
instance = X_test.iloc[0].values
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=10,
    num_samples=5000
)

# Show explanation
explanation.show_in_notebook()

# Get feature contributions as a list of (feature, weight) tuples
print(explanation.as_list())
```

**LIME for text classification:**

```python
from lime.lime_text import LimeTextExplainer

text_explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
exp = text_explainer.explain_instance(
    text_instance="The product exceeded my expectations",
    classifier_fn=vectorizer_and_model_pipeline.predict_proba,
    num_features=10
)
exp.show_in_notebook(text=True)
```

**SHAP vs. LIME:**

| Aspect | SHAP | LIME |
|---|---|---|
| Theoretical basis | Game theory (Shapley values) | Local linear approximation |
| Consistency | Always consistent | Can vary with sampling |
| Global explanations | Yes (aggregate SHAP) | No (local only) |
| Speed (trees) | Fast (TreeSHAP) | Slow (many model calls) |
| Speed (black-box) | Slow (KernelSHAP) | Moderate |
| Feature interactions | Captured | Not captured |
| Text/image support | Limited | Yes |

---

## Integrated Gradients

Integrated Gradients (IG) is an attribution method for differentiable models (neural networks). It computes the integral of gradients along a path from a baseline input to the actual input.

```
IG_i(x) = (x_i - x'_i) * integral[0→1] of (∂F(x' + α(x - x')) / ∂x_i) dα
```

Where `x'` is the baseline (e.g., all-zeros or all-black image) and `x` is the actual input.

```python
# Using Captum (PyTorch interpretability library)
from captum.attr import IntegratedGradients
import torch

model.eval()
ig = IntegratedGradients(model)

input_tensor = torch.tensor(X_test[:1], dtype=torch.float32, requires_grad=True)
baseline = torch.zeros_like(input_tensor)  # Baseline: all-zero input

attributions, delta = ig.attribute(
    inputs=input_tensor,
    baselines=baseline,
    target=1,  # Explain class 1 prediction
    n_steps=300,
    return_convergence_delta=True
)

print(f"Convergence delta: {delta.item():.6f}")  # Should be close to 0
print(f"Attribution magnitudes: {attributions.detach().numpy()}")
```

**Saturation Axiom:** IG satisfies completeness — attributions sum exactly to `F(x) - F(x')`, unlike plain gradients.

---

## Partial Dependence Plots and ICE Curves

### Partial Dependence Plots (PDP)

Show the marginal effect of one or two features on the model's prediction, averaging over the distribution of all other features:

```python
from sklearn.inspection import PartialDependenceDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1D PDP
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=["income", "age"],
    ax=axes
)
plt.suptitle("Partial Dependence Plots")
plt.tight_layout()

# 2D interaction PDP
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=[("income", "age")],  # Tuple for 2D
)
```

**Limitation:** PDPs show average effects and can be misleading when features are correlated (the averaged combinations may be unrealistic).

### Individual Conditional Expectation (ICE) Curves

ICE plots show the PDP for each individual sample, revealing heterogeneous effects:

```python
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=["income"],
    kind="both",  # "average" = PDP, "individual" = ICE, "both" = both
    subsample=200,  # Plot 200 random samples
    alpha=0.1  # Transparency for individual lines
)
```

---

## Model Cards

A Model Card is a short document accompanying a trained model that provides transparency about its intended use, performance, limitations, and ethical considerations. Introduced by Google in 2019.

**Standard sections:**
- Model details (architecture, training data, date, version)
- Intended use and out-of-scope uses
- Factors and disaggregated evaluation (performance by subgroup)
- Metrics and evaluation data
- Ethical considerations
- Caveats and recommendations

```python
# Using the model-card-toolkit library
from model_card_toolkit import ModelCardToolkit

mct = ModelCardToolkit()
model_card = mct.scaffold_assets()

model_card.model_details.name = "Fraud Detection Model v2.3"
model_card.model_details.overview = (
    "XGBoost classifier predicting transaction fraud probability. "
    "Trained on 12 months of transaction data."
)

# Add quantitative analysis
from model_card_toolkit.proto.model_card_pb2 import PerformanceMetric
metric = model_card.quantitative_analysis.performance_metrics.add()
metric.type = "AUC-ROC"
metric.value = "0.934"
metric.slice = "Overall"

mct.update_model_card(model_card)
mct.export_format(model_card=model_card, template_path="template.html")
```

---

## Regulatory Compliance and the EU AI Act

### EU AI Act (2024)

The EU AI Act classifies AI systems by risk level:

| Risk Level | Examples | Requirements |
|---|---|---|
| Unacceptable | Social scoring, real-time biometric surveillance | Prohibited |
| High | Credit scoring, hiring, medical diagnosis, law enforcement | Transparency, human oversight, explainability, data governance |
| Limited | Chatbots, deepfakes | Disclosure obligations |
| Minimal | Spam filters, AI in games | No specific obligations |

**Technical requirements for high-risk systems:**
- Logging of operations (audit trail)
- Accuracy, robustness, and cybersecurity measures
- Transparency documentation (technical documentation + instructions for use)
- Human oversight mechanisms
- Conformity assessment before market deployment

### GDPR Article 22 — Right to Explanation

When a decision is based solely on automated processing and produces legal or similarly significant effects, the data subject has the right to:
- Obtain human intervention
- Express their point of view
- Obtain an explanation of the decision
- Contest the decision

**Technical implementation:** Local explanations (SHAP waterfall, LIME) per prediction satisfy the spirit of the right to explanation. Store explanations alongside predictions in your serving logs.

### US Regulations

- **ECOA / Regulation B:** Adverse action notices must include specific reasons for credit denial
- **FCRA:** Consumers must be notified of adverse decisions based on credit reports
- **SR 11-7 (Fed guidance on model risk):** Requires conceptual soundness validation, including understanding of model drivers

---

## Interview Q&A

**Q1: What is the difference between global and local explainability? Give an example of each.**

Global explainability describes the overall behavior of a model across all predictions. Example: a SHAP summary plot showing that `income` is the most important feature overall. Local explainability describes why the model made a specific individual prediction. Example: a SHAP waterfall plot showing that for applicant #4521, a low credit score contributed -0.3 and high debt-to-income ratio contributed -0.2 to their denial.

**Q2: Why are SHAP values theoretically superior to permutation importance?**

SHAP values satisfy four game-theoretic axioms (efficiency, symmetry, dummy, linearity) that provide mathematical guarantees about fairness of attribution. Permutation importance suffers from: correlated features splitting importance arbitrarily, dependence on the choice of evaluation metric, and sensitivity to the number of permutation repeats. SHAP is also additive (individual SHAP values sum to the prediction), making it directly interpretable as contributions.

**Q3: What is the computational complexity of TreeSHAP vs. KernelSHAP?**

TreeSHAP runs in O(TLD²) — polynomial in the number of trees T, leaves L, and depth D. It can explain a prediction in milliseconds. KernelSHAP runs in O(2^M) in the worst case (exponential in M features), but in practice uses sampling, making it O(S × M) where S is the number of samples. For a model with 100 features, KernelSHAP might require minutes per prediction while TreeSHAP takes milliseconds.

**Q4: How does LIME ensure that local explanations are faithful to the black-box model?**

LIME weights perturbed samples by their proximity to the explained instance using an exponential kernel: `w(z) = exp(-D(x, z)² / σ²)`. Points closer to x get higher weight, so the sparse linear model is primarily fitted on the local neighborhood of x. However, faithfulness is not guaranteed — LIME is an approximation. The `num_samples` parameter controls the quality of the approximation.

**Q5: What is the baseline in Integrated Gradients and how do you choose it?**

The baseline is the reference input representing "absence of information" or "neutral input." Common choices: all-zeros for tabular/image data, the all-black image for vision, the all-[PAD] token sequence for text, or the mean training example. The choice significantly affects attributions — the baseline should represent a "no information" state for the problem domain. For text, the [MASK] token is often preferred over [PAD].

**Q6: How would you use SHAP to debug a data leakage problem?**

Train the model and compute SHAP values. If a feature with extremely high SHAP importance is one that should not be causally related to the target (e.g., a timestamp, an ID, or a feature computed after the prediction time), this signals leakage. Plot the SHAP dependence plot for that feature — if it shows an unnaturally perfect relationship, investigate the feature's data pipeline. TreeSHAP makes this fast enough to run in CI/CD after every training run.

**Q7: What is a partial dependence plot and when would you prefer ICE curves?**

A PDP shows the average marginal effect of a feature by averaging predictions over all other feature values. ICE curves show this effect for each individual data point. Prefer ICE when you suspect the relationship is heterogeneous — for example, if income increases prediction for some customers but decreases it for others (a crossing pattern in ICE reveals an interaction that the PDP would average away). PDPs can be misleading when features are highly correlated.

**Q8: Under GDPR, what does "right to explanation" require technically?**

The regulation does not mandate a specific technical method, but requires "meaningful information about the logic involved" in automated decisions. In practice: (1) store local explanations (e.g., top-N SHAP features) alongside each prediction in a log, (2) build an API to retrieve explanations for a given decision ID, (3) translate technical attributions into human-readable language (e.g., "Your application was primarily affected by your debt-to-income ratio (45%) which exceeds our threshold of 35%").

**Q9: How can you use SHAP for feature selection?**

Compute mean absolute SHAP values across the training set. Features with near-zero mean |SHAP| can be safely removed. This is more reliable than impurity-based importance because: it is not biased toward high-cardinality features, it accounts for interactions, and it directly measures impact on the model's output in interpretable units. You can also plot SHAP interaction values to identify redundant correlated features.

**Q10: What is a model card and what should it contain for a high-risk AI system?**

A model card is a brief transparency document for a model. For high-risk AI systems (under the EU AI Act), it should include: (1) model architecture and training details, (2) intended use cases and explicitly out-of-scope uses, (3) training data description including known biases, (4) performance metrics disaggregated by relevant subgroups (age, gender, race, geographic region), (5) fairness metrics and known failure modes, (6) ethical considerations and mitigation strategies, (7) human oversight requirements, and (8) versioning and update history.

**Q11: When would SHAP give misleading explanations?**

SHAP assumes feature independence when computing Shapley values (marginalizing over the feature distribution). When features are highly correlated, SHAP values can be misleading because the "absent features" in the coalition computation are filled with values from the joint distribution that never occur naturally. For example, if height and weight are correlated, SHAP may assign high importance to one and low to the other based on which was included in the coalition first, not reflecting the true marginal contribution.

---

## Common Pitfalls

**1. Confusing model explanations with causal explanations**
SHAP, LIME, and PDP describe the model's behavior, not the real-world causal structure. A high SHAP value for "zip code" explains the model's decision but does not mean zip code causes fraud. Never present model explanations as causal claims without a causal analysis.

**2. Trusting impurity-based importance for feature selection**
Impurity-based importance inflates the importance of high-cardinality features and is measured on training data, making it unreliable. Always use permutation importance or SHAP for reliable feature selection.

**3. Using KernelSHAP on large datasets without sampling**
KernelSHAP makes `nsamples` calls to the model per explained instance. For 10,000 test samples and `nsamples=500`, that is 5 million model calls. Always use TreeSHAP for tree models and only use KernelSHAP on a representative subsample.

**4. Ignoring SHAP interaction effects**
The main SHAP value for a feature includes its interaction effects with other features, which can be misleading. Use `shap.TreeExplainer(model, feature_perturbation="interventional")` and SHAP interaction values to decompose interaction effects explicitly.

**5. Presenting PDPs for correlated features**
A PDP for income marginalized over all other features may show effects for income levels that never co-occur with realistic values of correlated features (e.g., very high income with very high debt). Use Accumulated Local Effects (ALE) plots as a PDP alternative when features are correlated.

**6. Not checking LIME stability**
LIME uses random sampling, so explanations can vary between runs. Always run LIME multiple times and check consistency. If explanations change significantly across runs, increase `num_samples` or use SHAP instead.

**7. Applying explanation methods to the wrong model stage**
Always explain the model used in production, including preprocessing pipelines and post-processing steps. Explaining only the core model while ignoring imputation, scaling, or calibration gives an incomplete picture.

**8. Treating SHAP as ground truth for debugging**
SHAP explains what the model learned, not what is correct. A model that perfectly memorized spurious correlations will have high-confidence SHAP explanations pointing to irrelevant features. Always validate SHAP findings against domain knowledge.

---

## Related Topics

- [Intro to MLflow](intro_mlflow.md) — Tracking experiments and managing model lifecycle
- [Intro to Model Serving](intro_model_serving.md) — Deploying models and logging predictions for explanation storage
- [Intro to Data Quality](intro_data_quality.md) — Data drift detection and validation
- [Intro to Feature Stores](intro_feature_stores.md) — Managing features that feed into explainability analyses
- [Fairness and Bias in ML](../ml_concepts/fairness_bias.md) — Algorithmic fairness metrics and mitigation strategies
