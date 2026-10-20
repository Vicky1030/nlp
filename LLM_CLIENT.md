# 本地模型调用接口

`llm_client.py` 是 B 的公共调用层，供后续 Zero-shot 和 Few-shot 共用，不训练模型。已有试跑脚本和结果保留。模型只从本地读取，不联网下载；默认权重目录为本文件所在目录下的 `models`，需要 CUDA。

## Python 调用

在项目根目录使用 `.venv\Scripts\python.exe` 运行自己的脚本：

```python
from llm_client import LLMClient

client = LLMClient(log_path="outputs/my_calls.jsonl")
client.warmup()
response = client.generate(
    prompt="句子1：花呗怎么提高额度？\n句子2：如何提升花呗额度？",
    system_prompt="判断两个句子是否语义相似。相似输出1，不相似输出0。最终只输出0或1。",
    enable_thinking=False,
)
print(response.text)
print(response.prediction)
print(response.latency_ms)
```

只创建一次 `client`，循环调用 `client.generate(...)`；不要每个样本重新创建 client。Few-shot 可以在 prompt 中加入从 train 选择的带标签示例。当前待预测样本的真实 label 不应提供给模型。

开启思考模式：

```python
response = client.generate(
    prompt="句子1：花呗怎么提高额度？\n句子2：如何提升花呗额度？",
    system_prompt="判断语义是否相似，最终只输出0或1。",
    enable_thinking=True,
    max_new_tokens=2048,
    seed=42,
)
```

## 命令行试用

```powershell
cd D:\nlp
.\.venv\Scripts\python.exe llm_client.py
.\.venv\Scripts\python.exe llm_client.py --thinking
```

可使用 `--prompt`、`--system-prompt`、`--model-path`、`--max-new-tokens`、`--seed` 和 `--log` 指定参数。命令行默认先预热，再调用一次模型；日志追加到 `outputs/llm_client_calls.jsonl`。

## 返回结果

| 字段 | 含义 |
|---|---|
| text | 提取出的最终回答；思考未结束时为空字符串 |
| raw_output | 模型原始生成文本，包含思考段及答案 |
| prediction | 便于分类调用的严格 0/1 解析；无效或因长度截断时为 None，不能擅自改成 0 |
| thinking_complete | 思考结束为 True，思考未结束为 False，普通输出为 None |
| finish_reason | eos：正常结束；length：达到输出上限；stopped：其他停止 |
| input_tokens、output_tokens | 单次实际输入、输出 token 数，输出计数包含结束 token |
| latency_ms | 同步 GPU 后仅计 model.generate 耗时，与 C 现有预测脚本的计时范围接近 |
| request_latency_ms | 包含模板构建、分词、生成、解码及失败重试等待的请求耗时；排除模型加载和日志写入 |
| attempts | 实际尝试次数，1 表示没有重试 |
| model_id、model_fingerprint | 模型名称和实际文件的内容指纹 |
| settings | 本次思考、输出预算、采样和种子参数 |

`response.to_dict()` 可用于保存 JSON。模型输出没有强制限定为数字，因此调用者可以自由设计提示；分类便利字段 prediction 仅严格接受最终答案 0/1，不从解释中搜索数字。

普通模式默认贪心生成、最多 8 token；思考模式默认采样 temperature=0.6、top_p=0.95、top_k=20，最多 2048 token。可显式传入 do_sample、temperature、top_p、top_k。参数不同的实验需完整记录，不把效果差异完全归因于思考开关。

## 重试、日志和版本

默认最多重试一次，仅针对超时、连接失败、EAGAIN/EINTR/ETIMEDOUT 这类可重试错误。CUDA OOM、参数错误、模型配置错误直接抛出；无效生成是一个结果，不触发重新采样。不要用反复采样直到输出正确的方式筛选结果。

JSONL 日志记录 model_loaded、generation 和 generation_error 事件，包含输入提示、原始输出和实际参数。日志写入失败会抛出异常，不静默丢日志。此日志不是 D 的七字段预测文件；后续 Zero-shot/Few-shot 模块再按 D 的约定组织预测。

模型初始化时计算权重、配置和分词器文件的 SHA-256，再合成 model_fingerprint。因此首次创建 client 会多花一些时间；同一个 client 后续调用不重复加载或计算哈希。这个指纹用于标识实际本地文件，不冒充发布方的提交版本或官方哈希验证。

每次请求使用给定种子，调用后恢复外部 CPU/当前 GPU 的随机数状态。采样结果仍可能受硬件、库版本及运行方式影响，不能承诺跨环境逐位一致。接口面向单进程顺序调用，当前不提供批量或并发请求。
