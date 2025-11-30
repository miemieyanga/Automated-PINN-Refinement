# LLM-Driven PINN Automation Framework

A closed-loop multi-agent system that autonomously optimizes Physics-Informed Neural Networks (PINNs) to solve differential equations (ODEs/PDEs). It leverages Vision-Language Models (VLMs) to "see" training results and iteratively refine hyperparameters.

## ✨ Key Features

* **🤖 Multi-Agent Loop**: Autonomous cycle of reasoning, code generation, execution, and reflection.
* **👁️ Multimodal Evaluation**: Uses Vision-LLMs (e.g., GPT-4o) to diagnose training health by inspecting both **solution plots** and **loss curves** (detects trivial solutions, spectral bias, oscillation).
* **🛡️ Safety Sandbox (toy)**: Secure local execution with double-layer protection (LLM semantic checks + Regex blockers for `os.system`/`exec`).
* **Hybrid Generation**: Combines LLM reasoning with deterministic PyTorch templates to ensure code stability.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy matplotlib scipy openai pandas python-dotenv
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run Optimization

```bash
python main.py --task burgers_1d
```

## 📝 Supported Tasks

## Supported Tasks

| Task           | Description                                   | Difficulty                          |
|----------------|-----------------------------------------------|--------------------------------------|
| `burgers_1d`   | Fluid mechanics PDE with shock waves          | High     |
| `pendulum`     | Non-linear simple pendulum                    | Medium                               |
| `poisson_1d`   | Basic 1D Poisson equation                     | Low                                  |
