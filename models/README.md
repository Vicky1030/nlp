# 模型文件放置说明

基础模型不提交到 Git。请把固定版本的 `Qwen/Qwen3-1.7B` 下载到：

```text
models/qwen3-1.7b/
```

下载完成后，该目录至少应包含 `config.json`、tokenizer 文件和模型权重。根目录
`README.md` 提供了固定 revision 的下载命令。

最终 LoRA Adapter 已随仓库放在：

```text
models/lora_r16_a32_30899/
```

不要把虚拟环境、Qwen 基础模型或训练 checkpoint 提交到仓库。
