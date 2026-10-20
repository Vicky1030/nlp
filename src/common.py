"""共享的任务指令：B 的提示和 C 的训练/推理须严格统一。"""

MODEL_ID = "Qwen/Qwen3-1.7B"
TASK_INSTRUCTION = (
    "请判断下面两个中文句子表达的意思是否相同。"
    "相同输出1，不同输出0。只能输出一个数字，不要解释。"
)


def make_messages(sentence1: str, sentence2: str):
    question = (
        f"{TASK_INSTRUCTION}\n"
        f"句子1：{sentence1}\n"
        f"句子2：{sentence2}"
    )
    return [{"role": "user", "content": question}]
