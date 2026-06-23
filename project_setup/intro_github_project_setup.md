# How to Set Up a Project on GitHub — Complete Tutorial (2026 Edition)

A practical, end-to-end walkthrough for taking an ML/AI project from an empty
folder to a clean, collaborative, CI-backed GitHub repository. This is the guide
interviewers expect you to *already know* — version control hygiene, branching,
pull requests, and CI/CD are table stakes for ML Engineer, AI Engineer, and
MLOps roles.

> **Why this matters in interviews:** "Walk me through how you'd structure and
> ship a new ML project" is a common system/behavioral question. Being fluent
> with git, GitHub Actions, and a sane repo layout signals you can ship, not
> just prototype in notebooks.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1 — Install and Configure Git](#step-1--install-and-configure-git)
3. [Step 2 — Create the Repository](#step-2--create-the-repository)
4. [Step 3 — Initialize the Project Locally](#step-3--initialize-the-project-locally)
5. [Step 4 — The Essential Files](#step-4--the-essential-files)
6. [Step 5 — Connect Local to Remote](#step-5--connect-local-to-remote)
7. [Step 6 — Branching Strategy](#step-6--branching-strategy)
8. [Step 7 — Pull Requests & Code Review](#step-7--pull-requests--code-review)
9. [Step 8 — Protect the Main Branch](#step-8--protect-the-main-branch)
10. [Step 9 — Continuous Integration (GitHub Actions)](#step-9--continuous-integration-github-actions)
11. [Step 10 — Releases, Tags & Versioning](#step-10--releases-tags--versioning)
12. [Step 11 — GitHub Pages (Project Docs Site)](#step-11--github-pages-project-docs-site)
13. [Secrets & Security](#secrets--security)
14. [Authentication: SSH vs HTTPS vs gh CLI](#authentication-ssh-vs-https-vs-gh-cli)
15. [Common Mistakes](#common-mistakes)
16. [Interview Questions](#interview-questions)

---

## Prerequisites

| Tool | Why | Check |
|------|-----|-------|
| **Git** | Version control | `git --version` |
| **A GitHub account** | Remote hosting | github.com |
| **GitHub CLI (`gh`)** *(optional, recommended)* | Create repos/PRs from the terminal | `gh --version` |
| **An editor** | VS Code, JetBrains, etc. | — |

---

## Step 1 — Install and Configure Git

```bash
# macOS
brew install git
# Debian/Ubuntu
sudo apt-get install git
# Windows: download from git-scm.com

# One-time identity setup (shows up in your commits)
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"

# Sensible defaults
git config --global init.defaultBranch main      # default branch name
git config --global pull.rebase true             # cleaner history on pull
git config --global core.autocrlf input          # line-ending sanity (mac/linux)
```

> Verify with `git config --list`.

---

## Step 2 — Create the Repository

### Option A — On github.com (GUI)
1. Click **New repository**.
2. Name it (`my-ml-project`), add a one-line description.
3. Choose **Public** or **Private**.
4. Tick **Add a README**, **Add .gitignore → Python**, and **Choose a license** (MIT/Apache-2.0 are common).
5. **Create repository**, then clone:

```bash
git clone https://github.com/<you>/my-ml-project.git
cd my-ml-project
```

### Option B — From the terminal with `gh`
```bash
gh repo create my-ml-project --public --clone --gitignore Python --license mit
cd my-ml-project
```

### Option C — Push an existing local folder
See [Step 5](#step-5--connect-local-to-remote).

---

## Step 3 — Initialize the Project Locally

```bash
mkdir my-ml-project && cd my-ml-project
git init                      # creates the .git/ directory
python -m venv .venv          # isolated environment
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

> See the companion guide **[ML/AI Project Folder Structures](./intro_project_structure.md)**
> for the recommended directory layout to create here.

---

## Step 4 — The Essential Files

Every healthy repo has these. Skipping them is the #1 sign of an unmaintained project.

### `.gitignore`
Stop committing junk, secrets, and large artifacts. A minimal Python/ML version:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.env
.ipynb_checkpoints/

# ML artifacts (use DVC / model registry instead of git)
*.ckpt
*.pt
*.pth
*.onnx
data/raw/
data/processed/
models/
mlruns/
wandb/

# OS / editor
.DS_Store
.idea/
.vscode/
```

### `README.md`
The front door. Include: what it does, how to install, how to run, an example,
and a project structure overview. (See [Step 11](#step-11--github-pages-project-docs-site) to turn docs into a site.)

### `LICENSE`
No license = "all rights reserved" by default, meaning others legally cannot use
it. MIT (permissive) and Apache-2.0 (permissive + patent grant) are the usual
picks for open work.

### `requirements.txt` / `pyproject.toml`
Pin dependencies so the project is reproducible.

```bash
pip freeze > requirements.txt          # quick & dirty
# or, preferred for libraries/apps:
# define dependencies in pyproject.toml and use uv/poetry/pip-tools to lock
```

### `.env.example`
Commit a template of required env vars (**never the real `.env`**):

```bash
# .env.example  —  copy to .env and fill in
ANTHROPIC_API_KEY=
DATABASE_URL=
```

### `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`
For collaborative/open-source repos — set expectations for contributors.

---

## Step 5 — Connect Local to Remote

```bash
git add .
git commit -m "Initial project scaffold"

# Link to the GitHub remote you created in Step 2
git remote add origin https://github.com/<you>/my-ml-project.git

git branch -M main           # rename current branch to main
git push -u origin main      # -u sets the upstream so future pushes are just `git push`
```

> `git remote -v` confirms the remote URL.

---

## Step 6 — Branching Strategy

**Never commit straight to `main`.** Work on a branch, open a PR, merge after review.

### GitHub Flow (simplest — recommended for most ML projects)
```
main ──●──────────────●────────────● (always deployable)
        \            / \           /
         ●──●──●────●   ●──●──●────●
        feature/x      fix/y
```

```bash
git switch -c feature/add-training-pipeline   # create + switch (modern syntax)
# ...work, then...
git add .
git commit -m "feat: add training pipeline with early stopping"
git push -u origin feature/add-training-pipeline
```

### Branch naming conventions
| Prefix | Use for |
|--------|---------|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `experiment/` | ML experiments (`experiment/lr-sweep`) |
| `docs/` | Documentation only |
| `chore/` | Tooling, deps, CI |

### Conventional Commits
A widely-used commit message format that enables auto-changelogs and semantic versioning:

```
feat: add SHAP-based feature attribution
fix: correct off-by-one in sequence padding
docs: expand README quickstart
chore: bump torch to 2.4
refactor: extract dataloader into module
```

Format: `<type>(<optional scope>): <description>`.

---

## Step 7 — Pull Requests & Code Review

A **Pull Request (PR)** proposes merging your branch into `main` and is where
review, CI, and discussion happen.

```bash
# With the gh CLI
gh pr create --title "Add training pipeline" --body "Implements early stopping and checkpointing."

# Or push and click the link GitHub prints / open a PR in the web UI.
```

**A good PR:**
- Is **small and focused** (one logical change — easier to review, faster to merge).
- Has a clear title + description (what, why, how to test).
- Links the issue it closes: `Closes #42` in the body auto-closes the issue on merge.
- Passes CI (tests, lint) before requesting review.

**PR templates** — add `.github/pull_request_template.md` to standardize:

```markdown
## What
<!-- What does this PR do? -->

## Why
<!-- Why is it needed? -->

## How to test
<!-- Steps / commands -->

## Checklist
- [ ] Tests added/updated
- [ ] Docs updated
- [ ] CI passing
```

---

## Step 8 — Protect the Main Branch

In **Settings → Branches → Add branch protection rule** for `main`:
- ✅ Require a pull request before merging
- ✅ Require approvals (1+)
- ✅ Require status checks to pass (your CI)
- ✅ Require branches to be up to date before merging
- ✅ (Optional) Require signed commits / linear history

This guarantees nothing untested or unreviewed reaches `main`.

---

## Step 9 — Continuous Integration (GitHub Actions)

CI runs automatically on every push/PR. Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest ruff

      - name: Lint
        run: ruff check .

      - name: Run tests
        run: pytest -q
```

> For a deeper treatment (matrices, caching, Docker builds, deployment), see
> the **[GitHub Actions guide](../devops/intro_github_actions.md)**.

**Pre-commit hooks** catch issues *before* they hit CI. Add `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
      - id: ruff-format
```

```bash
pip install pre-commit && pre-commit install
```

---

## Step 10 — Releases, Tags & Versioning

Use **Semantic Versioning** (`MAJOR.MINOR.PATCH`):
- **MAJOR** — breaking changes
- **MINOR** — new, backward-compatible features
- **PATCH** — backward-compatible bug fixes

```bash
git tag -a v1.0.0 -m "First stable release"
git push origin v1.0.0
```

Then **Releases → Draft a new release** on GitHub (or `gh release create v1.0.0 --generate-notes`)
to attach changelogs and artifacts.

---

## Step 11 — GitHub Pages (Project Docs Site)

Turn `README.md`/`docs/` into a hosted website for free.

1. **Settings → Pages**.
2. **Source**: Deploy from a branch → `main` → `/ (root)` or `/docs`.
3. Save. Your site appears at `https://<you>.github.io/<repo>/`.

> This very repository is published with GitHub Pages — `index.html` at the root
> renders the markdown guides into a browsable site. A `.nojekyll` file disables
> Jekyll processing so files are served as-is.

For richer docs, use **MkDocs** (`mkdocs-material`) or **Sphinx** with a Pages
deploy workflow.

---

## Secrets & Security

- **Never commit secrets** (API keys, tokens, passwords). Use `.env` (git-ignored)
  locally and **GitHub Actions Secrets** (`Settings → Secrets and variables → Actions`)
  in CI: reference as `${{ secrets.ANTHROPIC_API_KEY }}`.
- **If you leak a secret**, rotate it immediately — git history is permanent.
  Removing it from the latest commit is not enough; revoke and reissue the key.
- Enable **Dependabot** (`Settings → Code security`) for automated dependency
  vulnerability alerts and PRs.
- Enable **secret scanning** and **push protection** to block accidental commits
  of credentials.

---

## Authentication: SSH vs HTTPS vs gh CLI

| Method | Setup | Best for |
|--------|-------|----------|
| **HTTPS + token** | Generate a Personal Access Token (PAT), use as password | Simple, works everywhere |
| **SSH keys** | `ssh-keygen` → add public key to GitHub | No password prompts, dev machines |
| **`gh auth login`** | One command, browser flow | Easiest; manages credentials for you |

```bash
# SSH key (one-time)
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub        # paste into GitHub → Settings → SSH keys
git remote set-url origin git@github.com:<you>/my-ml-project.git
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Committing `data/`, `models/`, `.venv/` | Add to `.gitignore`; use DVC / a model registry for large artifacts |
| Committing secrets | `.env` git-ignored + Actions Secrets; rotate if leaked |
| One giant PR | Split into small, focused PRs |
| Working directly on `main` | Use feature branches + branch protection |
| No README/LICENSE | Add both — they signal a real, usable project |
| Vague commit messages (`"fixed stuff"`) | Use Conventional Commits |
| No CI | Add a GitHub Actions workflow (Step 9) |

---

## Interview Questions

1. **Walk me through how you'd set up a new ML project repo.** → Init, venv,
   `.gitignore`, README/LICENSE, dependency pinning, branch protection, CI.
2. **`git merge` vs `git rebase`?** → Merge preserves history with a merge commit;
   rebase replays commits for a linear history. Don't rebase shared/public branches.
3. **How do you keep large model files / datasets out of git?** → `.gitignore` +
   DVC / Git LFS / a model registry (MLflow, S3) — git is for code, not binaries.
4. **What goes in CI for an ML repo?** → Lint, unit tests, data/schema validation,
   maybe a tiny training smoke test and model-eval gate.
5. **How do you handle secrets across local dev and CI?** → `.env` (git-ignored)
   locally, GitHub Actions Secrets in CI, never hardcoded.
6. **What is a protected branch and why use it?** → A branch that requires PRs,
   reviews, and passing checks before merge — prevents unreviewed/broken code on `main`.
7. **Explain trunk-based vs GitHub Flow vs Git Flow.** → Trunk-based: short-lived
   branches off `main`, frequent integration. GitHub Flow: branch → PR → merge.
   Git Flow: long-lived `develop`/`release` branches (heavier; less common for ML).

---

### Related Guides
- [ML/AI Project Folder Structures](./intro_project_structure.md)
- [GitHub Actions (CI/CD)](../devops/intro_github_actions.md)
- [Docker](../devops/intro_docker.md) · [MLflow](../mlops/intro_mlflow.md) · [LLMOps / MLOps Engineering](../mlops/intro_llmops_mlops_engineering.md)
