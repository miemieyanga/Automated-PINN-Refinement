# agents/hyperparam_agent.py
import json, re
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-mini"

HP_CHOICES = {
    "hidden_layers": [2, 3, 4, 5],
    "hidden_width": [32, 64, 128],
    "activation": ["tanh", "gelu", "relu"],
    "optimizer": ["adam", "lbfgs"],
    "lr": [5e-3, 1e-3, 5e-4],
    "epochs": [2000, 5000, 10000, 15000, 20000],
    "pde_collocation": [128, 512, 1024, 2048],
    "bc_weight": [1.0, 10.0, 50.0, 100.0],
    "lr_scheduler_type": ["step", "cosine", "none", "exponential"],
    "lr_decay_gamma": [0.1, 0.5, 0.9, 0.95, 0.99, 0.999]
}

def ask(system, user):
    r = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    return r.choices[0].message.content

class HyperparamAgent:
    @staticmethod
    def run(task: str, feedback: str = None):
        
        sys = "You are a PINN optimization expert. Output ONLY valid JSON."
        
        if feedback:
            instruction = (
                f"PREVIOUS FEEDBACK: {feedback}\n"
                f"INSTRUCTION: Specifically adjust parameters to FIX the issues mentioned above."
            )
        else:
            instruction = "INSTRUCTION: Select a robust set of initial parameters."

        msg = (
            f"Task: {task}\n"
            f"{instruction}\n"
            f"Choices: {json.dumps(HP_CHOICES, indent=2)}"
        )
        
        out = ask(sys, msg)
        
        m = re.search(r"\{.*\}", out, re.S)
        hp = json.loads(m.group(0)) if m else {}
        
        # --- 增强的兜底逻辑 (Robust Fallback) ---
        final_hp = {}
        for k, v_options in HP_CHOICES.items():
            raw_val = hp.get(k)
            
            # 尝试修复类型不匹配 (比如 int vs str)
            matched = False
            
            # 1. 直接匹配
            if raw_val in v_options:
                final_hp[k] = raw_val
                matched = True
            
            # 2. 尝试类型转换匹配 (针对 epochs, layers 等数字)
            if not matched and raw_val is not None:
                for opt in v_options:
                    # 如果选项是数字，尝试把 LLM 输出转数字
                    if isinstance(opt, (int, float)):
                        try:
                            if float(raw_val) == float(opt):
                                final_hp[k] = opt
                                matched = True
                                break
                        except:
                            pass
                    # 如果选项是字符串，尝试忽略大小写
                    elif isinstance(opt, str):
                        if str(raw_val).lower() == opt.lower():
                            final_hp[k] = opt
                            matched = True
                            break
            
            # 3. 仍然不匹配，才使用默认值
            if not matched:
                print(f"WARNING: Param '{k}' value '{raw_val}' invalid. Resetting to default: {v_options[0]}")
                final_hp[k] = v_options[0]
                
        return final_hp