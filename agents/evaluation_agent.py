# agents/evaluation_agent.py
import base64
import os
from openai import OpenAI

client = OpenAI()
# 【重要】确保使用支持视觉的模型，如 gpt-4o。如果 gpt-5-mini 不支持，请替换。
MODEL = "gpt-4o" 

def encode_image(image_path):
    """将本地图像文件编码为 Base64 字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

class EvaluationAgent:
    @staticmethod
    def run(metrics: dict):
        """
        接收指标字典，如果包含 image_path，则进行多模态评估。
        """
        sys_prompt = (
                    "You are a scientific evaluator. Analyze the results based on:"
                    "1. Numerical Metrics: especially MSE (Mean Squared Error) and MAE. Low MSE (<1e-3) is good."
                    "2. Visual Inspection: Compare the 'Ground Truth' vs 'PINN Prediction' curves in the plot."
                    "Combine these to judge if the model has converged to the correct physical solution."
                )
        
        image_path = metrics.get("image_path")
        base64_image = None
        if image_path:
            base64_image = encode_image(image_path)

        # 构造用户消息
        user_content = []
        # 1. 添加文本指标
        text_msg = f"Numerical Metrics provided:\n{metrics}\n"
        if base64_image:
            text_msg += "\nPlease also evaluate the attached plot showing Truth vs. Prediction."
        
        user_content.append({"type": "text", "text": text_msg})

        # 2. 如果有图像，添加图像内容
        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    # OpenAI API 要求的格式
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high" # 可选：low, high, auto
                }
            })
        elif image_path and not base64_image:
             user_content[0]["text"] += "\n(Warning: Image file listed in metrics was not found.)"

        try:
            # 发送请求
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=300 # 限制回复长度
            )
            comment = r.choices[0].message.content.strip()
        except Exception as e:
            comment = f"LLM Evaluation failed: {str(e)}"

        return {"metrics": metrics, "llm_comment": comment}