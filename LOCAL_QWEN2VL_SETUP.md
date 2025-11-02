# Qwen2VL 本地实现设置说明

本文档说明如何安装和测试使用本地 Qwen2VL 实现的 lmms-eval。

## 修改内容总结

已将 Qwen2VL 模型的实现本地化，主要修改包括：

1. **模型导入修改** (`lmms_eval/models/simple/qwen2_vl.py`)
   - 从 `transformers` 改为从本地 `lmms_eval.models.local_models.qwen2_vl` 导入
   - 使用本地的 `Qwen2VLForConditionalGeneration` 和 `Qwen2VLProcessor`

2. **添加验证 print** (`lmms_eval/models/local_models/qwen2_vl/modeling_qwen2_vl.py`)
   - 在 `Qwen2VLForConditionalGeneration.forward()` 方法中添加了明显的 print 语句
   - 每次前向传播都会打印：`🔥 Using LOCAL Qwen2VL implementation from lmms_eval/models/local_models/qwen2_vl/ 🔥`

3. **修复导入路径** (所有 `local_models/qwen2_vl/` 下的文件)
   - 将相对导入 (`from ...xxx`) 改为绝对导入 (`from transformers.xxx`)
   - 确保在 lmms-eval 包外也能正常工作

## 在有 GPU 机器上的操作步骤

### 1. 安装 lmms-eval（开发模式）

```bash
cd /path/to/lmms-eval
pip install -e .
```

**注意**：使用 `-e` 参数（editable mode）安装，这样修改代码后不需要重新安装。

### 2. 安装必要的依赖

确保已安装所需的包：

```bash
pip install transformers torch accelerate qwen-vl-utils decord pillow numpy
```

如果需要使用 flash attention：

```bash
pip install flash-attn --no-build-isolation
```

### 3. 运行测试（验证本地实现）

运行你之前的评测命令：

```bash
lmms-eval \
    --model qwen2_vl \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_mme \
    --output_path ./logs/
```

### 4. 验证本地实现是否被使用

**预期结果**：
- 在运行过程中，你应该会看到大量的输出：
  ```
  🔥 Using LOCAL Qwen2VL implementation from lmms_eval/models/local_models/qwen2_vl/ 🔥
  ```
- 这证明确实使用了本地实现，而不是 transformers 库中的实现

**对比结果**：
- 记录本地实现的测试结果（例如：MME 分数）
- 与之前使用 transformers 库的结果对比（你提到的 51.33）
- 结果应该相同或非常接近（可能有微小的浮点数差异）

### 5. 修改推理过程

现在你可以修改 `lmms_eval/models/local_models/qwen2_vl/modeling_qwen2_vl.py` 中的代码来改变推理行为：

- **修改 forward 方法**：在 line ~1775 开始
- **修改 generate 方法**：继承自 `GenerationMixin`
- **修改其他组件**：vision encoder、attention 机制等

每次修改后，由于使用了 `-e` 安装模式，直接重新运行评测即可，无需重新安装。

## 故障排查

### 问题 1：ImportError 相关错误

如果遇到导入错误，确保：
1. 已经用 `pip install -e .` 安装了 lmms-eval
2. transformers 版本足够新（建议 >= 4.37.0）

### 问题 2：没有看到验证 print

如果没有看到 `🔥 Using LOCAL...` 的输出：
1. 检查是否真的使用了本地安装的 lmms-eval：
   ```bash
   python -c "import lmms_eval; print(lmms_eval.__file__)"
   ```
   应该指向你的本地路径
2. 尝试清理缓存：
   ```bash
   pip uninstall lmms-eval
   pip install -e .
   ```

### 问题 3：结果差异很大

如果本地实现的结果与原始结果差异很大：
1. 检查是否所有文件都正确修改了导入
2. 检查 transformers 版本是否一致
3. 检查随机种子设置

## 下一步

验证通过后，你可以：
1. 移除或注释掉验证 print（line 1775-1777），避免日志过多
2. 开始修改推理逻辑，实现你的研究想法
3. 每次修改后运行评测，观察结果变化

## 文件位置参考

- **主要模型文件**：`lmms_eval/models/local_models/qwen2_vl/modeling_qwen2_vl.py`
- **配置文件**：`lmms_eval/models/local_models/qwen2_vl/configuration_qwen2_vl.py`
- **图像处理**：`lmms_eval/models/local_models/qwen2_vl/image_processing_qwen2_vl.py`
- **评测模型类**：`lmms_eval/models/simple/qwen2_vl.py`

祝研究顺利！🚀


