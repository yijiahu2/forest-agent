import os
import json
from openai import OpenAI


def get_doubao_client():
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY is not set in environment.")

    return OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


def call_doubao_json(prompt: str) -> dict:
    model = os.environ.get("ARK_MODEL")
    if not model:
        raise ValueError("ARK_MODEL is not set in environment.")

    client = get_doubao_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是森林单木分割参数优化智能体。"
                    "你必须只输出合法JSON，不要输出解释性前缀、markdown代码块或额外文字。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content.strip()

    # 去掉可能的 markdown 包裹
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"Doubao output is not valid JSON:\n{content}\n\nError: {e}")