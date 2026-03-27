# Contributing to CrackingMachineLearningInterview

Thank you for helping make this repository better! Contributions of all kinds are welcome — new questions, improved answers, new guides, code examples, and bug fixes.

---

## Table of Contents
1. [Ways to Contribute](#ways-to-contribute)
2. [Getting Started](#getting-started)
3. [Repository Structure](#repository-structure)
4. [Content Guidelines](#content-guidelines)
5. [Formatting Standards](#formatting-standards)
6. [Review Process](#review-process)
7. [Code of Conduct](#code-of-conduct)

---

## Ways to Contribute

| Type | Description | Examples |
|------|-------------|---------|
| **New Q&A** | Add interview questions and answers | Classic ML Q&A in README.md |
| **Improve answers** | Expand or clarify existing answers | Add code examples, fix inaccuracies |
| **New guide** | Create a new topic guide | New framework or algorithm coverage |
| **Code examples** | Add working Python code to guides | Scikit-learn, PyTorch, HuggingFace examples |
| **Bug fixes** | Fix typos, broken links, incorrect info | Spelling errors, outdated info |
| **Translations** | Translate content to other languages | Spanish, Chinese, etc. |

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/CrackingMachineLearningInterview.git
cd CrackingMachineLearningInterview

# Add the upstream remote
git remote add upstream https://github.com/shafaypro/CrackingMachineLearningInterview.git
```

### 2. Create a Branch

```bash
# Always create a feature branch — never commit directly to main
git checkout -b add-transformer-architecture-guide
# or
git checkout -b fix-regex-typo-readme
```

Branch naming conventions:
- `add-<topic>` — for new content
- `fix-<description>` — for bug fixes
- `improve-<topic>` — for expanding existing content
- `update-<topic>` — for updating outdated content

### 3. Make Your Changes

Follow the content and formatting guidelines below. Test that all links work.

### 4. Commit Your Changes

```bash
git add your-changed-files.md
git commit -m "Add transformer architecture guide with attention mechanism Q&A"
```

Write clear commit messages:
- Use present tense: "Add X" not "Added X"
- Be specific: "Add BERT fine-tuning examples to NLP guide" not "Update NLP"
- Reference issues: "Fix broken link in RAG guide (closes #42)"

### 5. Open a Pull Request

```bash
git push origin your-branch-name
```

Then open a Pull Request on GitHub. Use the PR template:
- **What**: Describe what you added or changed
- **Why**: Explain the motivation
- **Preview**: Link to the specific section if adding Q&A

---

## Repository Structure

```
CrackingMachineLearningInterview/
├── README.md                        ← Classic ML Q&A (169+ questions)
├── CONTRIBUTING.md                  ← This file
├── docs/
│   ├── 2026-interview-roadmap.md    ← Modern interview topics overview
│   ├── 2026-additional-questions.md ← Modern Q&A (25+ questions)
│   ├── study-pattern.md             ← Study guide with difficulty levels
│   └── resources-and-references.md ← Books and external resources
├── ai_genai/                        ← AI/GenAI/LLM topics
│   ├── intro_rag.md
│   ├── intro_vector_databases.md
│   └── ...
├── classical_ml/                    ← Classical ML algorithms
│   ├── intro_clustering.md
│   ├── intro_time_series.md
│   └── ...
├── mlops/                           ← MLOps and production ML
│   ├── intro_mlflow.md
│   ├── intro_model_serving.md
│   └── ...
├── cloud_ml/                        ← Cloud ML platforms
│   └── intro_cloud_ml_platforms.md
├── data_engineering/                ← Data engineering tools
│   ├── intro_apache_spark.md
│   └── ...
└── devops/                          ← DevOps and infrastructure
    ├── intro_docker.md
    └── ...
```

### Where to Add New Content

| Content Type | Location |
|-------------|---------|
| Classic ML interview questions | `README.md` under relevant section |
| Modern AI/LLM questions | `docs/2026-additional-questions.md` |
| New AI/GenAI guide | `ai_genai/intro_<topic>.md` |
| New classical ML guide | `classical_ml/intro_<topic>.md` |
| New MLOps guide | `mlops/intro_<topic>.md` |
| New data engineering guide | `data_engineering/intro_<topic>.md` |
| New DevOps guide | `devops/intro_<topic>.md` |
| New cloud platform guide | `cloud_ml/intro_<topic>.md` |

---

## Content Guidelines

### What Makes a Good Interview Question/Answer?

**Good questions:**
- Appear frequently in real ML interviews
- Test both conceptual understanding and practical application
- Have clear, unambiguous answers
- Are at the right difficulty level (Beginner/Intermediate/Advanced)

**Good answers:**
- Start with a clear 1-2 sentence definition
- Include a concrete example or analogy
- Address the "why" (motivation) and the "when" (use cases)
- Include code where it adds clarity
- Mention tradeoffs and limitations
- Are 3-10 sentences for simple questions; more for complex ones

**Bad answers to avoid:**
- Single-sentence answers without explanation
- Copy-pasted Wikipedia text
- Answers that are technically correct but miss practical nuance
- Answers without examples

### Example of a Good Q&A

```markdown
#### What is the difference between precision and recall?

Precision measures what fraction of your positive predictions were correct:
Precision = TP / (TP + FP). It answers: "Of everything I labeled positive, how many were actually positive?"

Recall measures what fraction of actual positives you found:
Recall = TP / (TP + FN). It answers: "Of all actual positives, how many did I find?"

There is always a tradeoff: increasing the classification threshold raises precision but reduces recall. Business context dictates the priority:
- Fraud detection: prioritize recall (catch most fraudulent transactions, even with false alarms)
- Email spam filter: prioritize precision (don't mark legitimate emails as spam)
- Medical diagnosis (cancer): prioritize recall (better to investigate false positives than miss true positives)

The F1 score is the harmonic mean: F1 = 2 * (precision * recall) / (precision + recall), providing a single balanced metric.
```

### Guide Structure Requirements

New topic guides must include:

1. **Title and introduction** — what the technology is and why it matters
2. **Table of Contents** — linked to anchors
3. **Core concepts** — definitions and diagrams/tables
4. **Code examples** — working Python code with comments
5. **Comparison tables** — when relevant (vs alternatives)
6. **Interview Q&A section** — minimum 8 questions with detailed answers
7. **Common Pitfalls section** — table format with problem + fix
8. **Related Topics section** — links to related guides in the repo

---

## Formatting Standards

### Markdown Conventions

```markdown
# Top-level title (one per file)
## Major section
### Subsection

**Bold** for key terms and emphasis
`code` for inline code, file names, variable names
*italics* for book/paper titles
```

### Code Blocks

Always specify the language:

```python
# Python code with descriptive comments
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

```bash
# Shell commands
pip install scikit-learn
```

```sql
-- SQL examples
SELECT user_id, COUNT(*) as purchase_count
FROM orders
GROUP BY user_id
```

### Tables

Use tables for comparisons and summaries:

```markdown
| Method | Pros | Cons | Use When |
|--------|------|------|---------|
| L1 (Lasso) | Feature selection | Unstable with correlated features | Many irrelevant features |
| L2 (Ridge) | Stable, all features used | No feature selection | Features all potentially relevant |
```

### Difficulty Tags

Tag Q&A with difficulty level:
- 🟢 Beginner — conceptual, expected from all candidates
- 🟡 Intermediate — applied, 2-5 YOE
- 🔴 Advanced — deep technical, senior roles

---

## Review Process

1. **Automated checks**: All PRs must have valid Markdown (no broken syntax)
2. **Content review**: A maintainer reviews for accuracy and completeness
3. **Formatting review**: Check adherence to style guide
4. **Merge**: Squash-merge to keep clean history

**Timeline**: We aim to review PRs within 5-7 business days. If you haven't heard back in 10 days, feel free to ping in the PR comments.

### Review Checklist (for reviewers)

- [ ] Content is technically accurate
- [ ] Code examples work as written
- [ ] Links are valid and point to correct locations
- [ ] Follows formatting standards
- [ ] Q&A answers are sufficiently detailed (not just 1-2 sentences)
- [ ] No duplicate content with existing guides
- [ ] New guide is linked from README.md

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to uphold its standards.

Key points:
- Be respectful and constructive in all interactions
- Welcome contributions from all skill levels
- Provide feedback that helps contributors improve
- Credit others' work appropriately

---

## Questions?

- Open a [GitHub Issue](https://github.com/shafaypro/CrackingMachineLearningInterview/issues) for questions or suggestions
- For major structural changes, open an issue first to discuss before implementing

Thank you for contributing! Every improvement helps someone prepare for their next ML interview.
