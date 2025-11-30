# main.py
import json
import argparse
import traceback
from typing import Dict, Any, Optional
from template import PINN_TEMPLATE
from problems import PROBLEMS
import types
from dataclasses import dataclass
from agents.hyperparam_agent import HyperparamAgent
from agents.format_agent import FormatAgent
from agents.safety_agent import SafetyAgent
from agents.execution_agent import ExecutionAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reflection_agent import ReflectionAgent

# 0. 定义基准参数 (Baseline)
BASELINE_HP = {
    "hidden_layers": 3,
    "hidden_width": 128,
    "activation": "tanh",
    "optimizer": "adam",
    "lr": 1e-3,
    "epochs": 2000,
    "pde_collocation": 128,
    "bc_weight": 10.0,
    "lr_scheduler_type": "step",
    "lr_decay_gamma": 0.9
}

@dataclass
class PipelineOutput:
    hp: Dict[str, Any]
    code: str
    safety: Dict[str, Any]
    evaluation: Dict[str, Any]
    reflection: str
    plan: Dict[str, Any]

class Runner:
    @staticmethod
    def run_module(code: str, seed: int = 42, round_num: int = 0) -> Dict[str, Any]:
        mod = types.ModuleType("generated_pinn")
        exec(code, mod.__dict__)
        # 传递 round_num 给 template 里的函数
        return mod.train_and_evaluate(seed=seed, round_num=round_num)

# 修改 run_pipeline，增加 override_hp 参数
def run_pipeline(task_name: str, round_num: int, feedback: Optional[str] = None, override_hp: Optional[Dict] = None) -> PipelineOutput:
    # 0) 获取问题定义
    if task_name not in PROBLEMS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(PROBLEMS.keys())}")
    
    prob_def = PROBLEMS[task_name]
    task_description = prob_def["description"]

    # 1) 超参决策
    if override_hp:
        # 如果是 Baseline 轮，直接使用传入的参数，跳过 LLM 思考
        print(f"--> [Mode: BASELINE] Using fixed hyperparameters.")
        hp = override_hp
    else:
        # 否则让 Agent 思考
        print(f"--> [Mode: OPTIMIZE] Agent selecting params based on feedback...")
        hp = HyperparamAgent.run(task_description, feedback)

    # 2) 代码生成 (注入物理 + 注入超参)
    code_with_physics = PINN_TEMPLATE.replace("{input_dim}", str(prob_def["input_dim"])) \
                                     .replace("{output_dim}", str(prob_def["output_dim"])) \
                                     .replace("{domain_min}", str(prob_def["domain_range"][0])) \
                                     .replace("{domain_max}", str(prob_def["domain_range"][1])) \
                                     .replace("{physics_code}", prob_def["physics_code"]) \
                                     .replace("{boundary_code}", prob_def["boundary_code"]) \
                                     .replace("{plotting_code}", prob_def["plotting_code"])

    code = FormatAgent.run(hp, template=code_with_physics)

    # 3) 安全检查
    safety = SafetyAgent.run(code)
    if safety.get("blockers"):
        raise RuntimeError(f"Safety blockers: {json.dumps(safety['blockers'])}")

    # 3.5) 执行规划
    plan = ExecutionAgent.plan(hp)
    if not plan.get("proceed", True):
        evaluation = {"metrics": {}, "llm_comment": f"Execution skipped: {plan.get('notes','')}"}
        # 即使跳过，也可以反思一下参数为什么不好
        reflection = ReflectionAgent.run(hp, evaluation)
        return PipelineOutput(hp, code, safety, evaluation, reflection, plan)

    # 4) 本地执行
    try:
        metrics = Runner.run_module(code, seed=plan.get("seed", 42), round_num=round_num)
    except Exception as e:
        metrics = {"error": str(e), "trace": traceback.format_exc()}

    # 5) 评估 (这里会自动读取生成的图像进行多模态分析)
    evaluation = EvaluationAgent.run(metrics)

    # 6) 反思 (LLM 会基于 Baseline 的糟糕表现提出改进建议)
    reflection = ReflectionAgent.run(hp, evaluation)

    return PipelineOutput(hp, code, safety, evaluation, reflection, plan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="van_der_pol", choices=list(PROBLEMS.keys()))
    args = parser.parse_args()
    TASK_NAME = args.task

    # ===== Round 0: Initial Evaluation (Baseline) =====
    print(f"\n================ ROUND 0: BASELINE ({TASK_NAME}) ================\n")
    # 强制使用默认参数运行
    round0 = run_pipeline(TASK_NAME, round_num=0, override_hp=BASELINE_HP)
    
    print("\n--- Baseline Metrics ---\n", round0.evaluation.get("metrics"))
    print("\n--- Baseline Visual Analysis ---\n", round0.evaluation.get("llm_comment"))
    print("\n--- Baseline Reflection (Advice for Round 1) ---\n", round0.reflection)

    print("\n--- hp ---\n", round0.hp)

    # 构造反馈给 Round 1
    # 我们明确告诉 LLM：这是 Baseline 的结果，请基于此进行优化
    feedback_r0 = (
        f"CONTEXT: We ran a BASELINE simulation with generic parameters: {json.dumps(BASELINE_HP)}.\n"
        f"BASELINE RESULTS: {json.dumps(round0.evaluation.get('metrics'))}\n"
        f"BASELINE VISUAL ANALYSIS: {round0.evaluation.get('llm_comment')}\n"
        f"ADVICE FROM REFLECTION AGENT: {round0.reflection}\n\n"
        f"GOAL: Based on the failure/shortcomings of the baseline, select BETTER hyperparameters to solve the task."
    )

    # ===== Round 1: First Optimization =====
    print(f"\n================ ROUND 1: OPTIMIZATION ================\n")
    round1 = run_pipeline(TASK_NAME, round_num=1, feedback=feedback_r0)
    
    print("\n--- Round 1 Metrics ---\n", round1.evaluation.get("metrics"))
    print("\n--- Round 1 Visual Analysis ---\n", round1.evaluation.get("llm_comment"))
    print("\n--- Baseline Reflection (Advice for Round 2) ---\n", round1.reflection)
    print("\n--- hp ---\n", round1.hp)

    feedback_r1 = (
        f"PREVIOUS ATTEMPT (Round 1): {json.dumps(round1.hp)}\n"
        f"METRICS: {json.dumps(round1.evaluation.get('metrics'))}\n"
        f"VISUAL EVALUATION: {round1.evaluation.get('llm_comment')}\n"
        f"REFLECTION: {round1.reflection}"
    )
    
    print(f"\n================ ROUND 2: REFINEMENT ================\n")
    round2 = run_pipeline(TASK_NAME, round_num=2, feedback=feedback_r1)
    print("\n--- Round 2 Metrics ---\n", round2.evaluation.get("metrics"))
    print("\n--- Baseline Reflection (Advice for Round 3) ---\n", round2.reflection)
    print("\n--- hp ---\n", round2.hp)

    feedback_r2 = (
        f"PREVIOUS ATTEMPT (Round 2): {json.dumps(round2.hp)}\n"
        f"METRICS: {json.dumps(round2.evaluation.get('metrics'))}\n"
        f"VISUAL EVALUATION: {round2.evaluation.get('llm_comment')}\n"
        f"REFLECTION: {round2.reflection}"
    )
    
    print(f"\n================ ROUND 3: REFINEMENT ================\n")
    round3 = run_pipeline(TASK_NAME, round_num=3, feedback=feedback_r2)
    print("\n--- Round 3 Metrics ---\n", round3.evaluation.get("metrics"))
    print("\n--- hp ---\n", round3.hp)

    print("\n✅ Optimization Complete.")