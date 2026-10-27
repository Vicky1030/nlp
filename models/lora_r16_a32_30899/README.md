# 最终 LoRA Adapter

这是 C 成员最终的 `rank=16 / alpha=32 / 30,899` AFQMC Adapter，来自仓库 C
分支的 `outputs/full_train_v1/qlora_r16_a32_30899/adapter/`。实时入口只需要：

- `adapter_config.json`
- `adapter_model.safetensors`

校验值：

```text
adapter_config.json          1043 bytes  SHA256 5561ef2f75c15c7670e0e7406fb5b3e84eec080c0bcbf71ebd05155473197ab2
adapter_model.safetensors 12859984 bytes  SHA256 04f7aca87749bbf9365beed880bf1ddd317522f315f7ca48bb36e39871ab147b
```

基础模型仍需按根目录 `README.md` 下载到 `models/qwen3-1.7b/`。
