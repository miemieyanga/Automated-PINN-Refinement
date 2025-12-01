# agents/reflection_agent.py
import json
from openai import OpenAI
from agents.hyperparam_agent import HP_CHOICES

client = OpenAI()
MODEL = "gpt-5-mini"

class ReflectionAgent:
    @staticmethod
    def run(hp: dict, eval_out: dict):
        choices_str = json.dumps(HP_CHOICES, indent=2)

        # --- 精简且开放的 System Prompt ---
        sys = f"""You are an expert PINN optimization coach. 
Diagnose the failure based on the Visual Description and Metrics (MSE/MAE). Give your suggestions. Don't ask any questions.

### Core PINN Principles:
1. **Stiffness & Trivial Solutions**: If prediction is a flat line or zero, the model is stuck. -> Suggest **LBFGS**, significantly higher `bc_weight`, or specific activations (`silu`/`tanh`).
2. **Spectral Bias**: If trends match but high-frequency wiggles are missing. -> Increase `hidden_layers`/`width`.
3. **Convergence**: If Loss stagnates. -> Suggest `lr_scheduler` or `lbfgs`.
4. **Generalization**: If Physics Loss is low but MSE is high. -> The model found a physical but wrong solution. -> Enforce BCs stronger or check domain normalization.

You must suggest hyperparameter updates STRICTLY from the following available choices:
{choices_str}

### Computational Cost Awareness (Time Consumption):
- **Expensive**: `optimizer="lbfgs"`, `epochs` > 10000, `pde_collocation` > 2000.
- **Cheap**: `optimizer="adam"`, `hidden_width` increases.
-> **Strategy**: Only suggest "Expensive" settings if the model is underfitting or failing (trivial solution). If performance is decent but slow, suggest cheaper alternatives.

### Goal:
Propose 3-5 CONCRETE, actionable hyperparameter updates for the next round. One stage training Only. 
Do not be limited to the above list; use your intuition. 
"""
        
        # --- 构造消息 ---
        metrics = eval_out.get("metrics", {})
        visual = eval_out.get("llm_comment", "No visual info.")
        
        msg = (
            f"Current Params: {json.dumps(hp, indent=2)}\n"
            f"Results: {json.dumps(metrics, indent=2)}\n"
            f"Visual Evaluation: \"{visual}\"\n\n"
            f"What specific parameters should I change to fix this? Return bullet points."
        )

        r = client.chat.completions.create(model=MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":msg}])
        
        return r.choices[0].message.content.strip()