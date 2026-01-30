# AutoSteer 🕵️‍♂️

### The Feature Hunter Agent for LLM Interpretability

AutoSteer is a mechanistic interpretability agent designed to **extract**, **validate**, and **steer** high-level features in Large Language Models (LLMs) using **Activation Addition**.

> **"We don't just find the feature; we prove it's a vector."**

---

## 🧬 The Philosophy (Rigorous Interrogation)

Unlike simple steering scripts, AutoSteer is built on a foundation of rigorous mechanistic inquiry. We do not blindly add vectors; we interrogate the model's internal physics.

### 1. The Mechanics

We intervene in the **Residual Stream** (`hook_resid_pre`).

- _Why?_ The residual stream is the "bandwidth" of the model. Injecting here effects a global state change, whereas injecting into attention heads only alters local token mixing.

### 2. The Linearity Assumption

We operate on the **Linear Representation Hypothesis** (Elhage et al., 2022).

- _Verification:_ We assume features like "Pirate Persona" exist as linear directions. AutoSteer verifies this by calculating `Mean(Positive) - Mean(Neutral)` and validating the vector's norm.

### 3. Safety via Normalization

We enforce **Unit Normalization**.

- _The Problem:_ Raw activation differences can have arbitrary magnitudes. Adding a vector with Norm=100 to a stream with Norm=10 destroys the signal.
- _The Solution:_ We normalize our steering vector $\hat{v} = v / \|v\|$ and introduce an explicit `STEERING_COEFF` (default: 5.0) to control injection strength precisely.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- `uv` package manager (recommended for speed) or `pip`.
- Apple Silicon (MPS) or NVIDIA GPU (CUDA) recommended.

### Installation

1.  **Clone & Setup**:
    ```bash
    git clone https://github.com/your-username/autosteer.git
    cd autosteer
    make setup
    ```

### Running the Agent

This runs `src/main.py`, which loads `gpt2-small`, calculates a "Pirate" steering vector, and generates steered text.

```bash
make run
```

**Expected Output:**

```text
--> [Init] Loading gpt2-small on mps...
--> [Analysis] Extracting vectors from 3 pairs...
--> [Success] Steering vector isolated and normalized.

--> [Gen] Prompt: 'I went to the grocery store and' | Steered: True
    Output: I went to the grocery store and... Arr matey! I plundered the snacks!
```

---

## 🔬 Interactive Deep Dive (Marimo)

For a visual interrogation of the model, run our local **Marimo** notebook. This is where you can "touch" the math.

```bash
uv run marimo edit notebook/deep_dive.py
```

**What you can explore:**

1.  **Mechanics**: Probe the residual stream magnitude.
2.  **Linearity**: Visualize "Pirate" vs "Neutral" clusters using **PCA**.
3.  **Normalization slider**: Real-time adjustment of standard deviation injection.
4.  **Layer Sweep**: Automatically find which layer has the strongest feature separation.

---

## 📂 Project Structure

```text
autosteer/
├── src/
│   └── main.py          # Core Agent (Production Logic)
├── notebook/
│   └── deep_dive.py     # Interactive Visual interrogation (Marimo)
├── .context/            # Project history & tracking
│   ├── changelog.md
│   └── summary.md
├── tests/               # Consistency checks
├── Makefile             # Automation
├── pyproject.toml       # Modern dependency management
└── PRD.md               # Product Requirements & Research Goals
```

---

## 📚 Citations & Theory

- **Activation Addition**: [Turner et al. (2023)](https://arxiv.org/abs/2308.10248) - _Activation Addition: Steering Language Models Without Optimization_.
- **Linear Representation**: [Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) - _Toy Models of Superposition_.
- **TransformerLens**: [Nanda (2022)](https://github.com/neelnanda-io/TransformerLens) - The library that makes this possible.

---

\_Built with ❤️ and scientific rigor by Uday Phalak
