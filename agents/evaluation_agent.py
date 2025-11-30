# agents/evaluation_agent.py
import base64
import os
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o" # 确保模型支持视觉

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

class EvaluationAgent:
    @staticmethod
    def run(metrics: dict):
        # 1. 获取两张图片的路径
        pred_path = metrics.get("image_path")
        loss_path = metrics.get("loss_image_path")
        
        b64_pred = encode_image(pred_path) if pred_path else None
        b64_loss = encode_image(loss_path) if loss_path else None

        # 2. 构造 System Prompt，加入对 Loss 的分析要求
        sys_prompt = (
            "You are a scientific evaluator utilizing visual inspection. "
            "You will receive TWO images:\n"
            "1. Prediction Plot: Comparison between Ground Truth (Blue) and PINN (Red).\n"
            "2. Loss Curve: Log-scale training loss over iterations.\n\n"
            "Your tasks:\n"
            "- Analyze the Prediction: Check for overlap, boundary errors, or smoothness.\n"
            "- Analyze the Loss Curve: Is it converging? Oscillating? Stuck (flat)? Or spiking?\n"
            "- Combine these to judge convergence quality. (e.g., 'Loss is low but prediction is flat' -> Trivial Solution)."
        )

        # 3. 构造 User Message
        user_content = []
        text_msg = f"Numerical Metrics:\n{metrics}\n"
        if not b64_pred: text_msg += "\n(Warning: Prediction plot missing)"
        if not b64_loss: text_msg += "\n(Warning: Loss plot missing)"
        
        user_content.append({"type": "text", "text": text_msg})

        # 添加预测图
        if b64_pred:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_pred}", "detail": "high"}
            })
            user_content.append({"type": "text", "text": "Image 1: Prediction vs Truth"})

        # 【新增】添加 Loss 图
        if b64_loss:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_loss}", "detail": "high"}
            })
            user_content.append({"type": "text", "text": "Image 2: Loss Curve"})

        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=400
            )
            comment = r.choices[0].message.content.strip()
        except Exception as e:
            comment = f"LLM Evaluation failed: {str(e)}"

        return {"metrics": metrics, "llm_comment": comment}