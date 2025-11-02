# 快速开始：使用本地 Qwen2VL 实现

## 🎯 目标

已将 Qwen2VL 模型实现本地化，现在可以修改推理过程并在评测中反映出来。

## ✅ 已完成的修改

1. ✅ 将所有 Qwen2VL 相关代码从 transformers 导入改为本地导入
2. ✅ 修复所有本地文件的导入路径（相对路径 → 绝对路径）
3. ✅ 在 `forward()` 方法中添加验证 print 语句
4. ✅ 创建测试脚本验证本地实现

## 🚀 在 GPU 机器上的操作步骤

### 步骤 1：安装（editable 模式）

```bash
cd /home/hqs123/class_code/LLM_Research/MLLM-Reasoning/lmms-eval
pip install -e .
```

### 步骤 2：快速验证

```bash
# 运行测试脚本
python test_local_qwen2vl.py
```

**期望看到**：
```
🔥 Using LOCAL Qwen2VL implementation from lmms_eval/models/local_models/qwen2_vl/ 🔥
```

### 步骤 3：完整评测验证

```bash
lmms-eval \
    --model qwen2_vl \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_local \
    --output_path ./logs/
```

**期望结果**：
- 看到大量 `🔥 Using LOCAL...` 输出
- MME 分数应该与之前相同/相近（~51.33）

### 步骤 4：开始修改

修改文件：
```
lmms_eval/models/local_models/qwen2_vl/modeling_qwen2_vl.py
```

关键位置：
- `forward()` 方法：第 1709 行开始
- `Qwen2VLForConditionalGeneration` 类：第 1666 行开始

## 📁 关键文件位置

| 文件 | 用途 |
|------|------|
| `lmms_eval/models/simple/qwen2_vl.py` | 评测入口（已改为导入本地实现）|
| `lmms_eval/models/local_models/qwen2_vl/modeling_qwen2_vl.py` | **主要修改文件**（模型实现）|
| `lmms_eval/models/local_models/qwen2_vl/configuration_qwen2_vl.py` | 配置类 |
| `lmms_eval/models/local_models/qwen2_vl/processing_qwen2_vl.py` | 处理器 |

## 🔧 修改推理流程示例

```python
# 在 modeling_qwen2_vl.py 的 forward 方法中（约 1779 行后）

def forward(self, ...):
    # 验证 print（可以在验证通过后删除）
    print("🔥 Using LOCAL Qwen2VL implementation...")
    
    # 原始代码
    output_attentions = ...
    output_hidden_states = ...
    
    # 🎯 在这里添加你的修改
    # 例如：修改 attention mask、改变推理逻辑等
    
    outputs = self.model(...)
    
    # ... 其余代码
```

## ⚠️ 注意事项

1. **使用 `-e` 安装**：这样修改代码后无需重新安装
2. **验证 print**：确认看到本地实现被使用后，可以删除/注释掉 print
3. **结果对比**：首次运行应该与原版结果一致，确保迁移成功
4. **依赖管理**：确保 transformers 版本一致（用于导入基类）

## 📚 详细文档

查看 `LOCAL_QWEN2VL_SETUP.md` 了解更多细节和故障排查。

## ✨ 完成标志

- [ ] 在 GPU 机器上运行 `pip install -e .`
- [ ] 运行 `test_local_qwen2vl.py` 看到验证输出
- [ ] 运行完整评测，结果与之前一致
- [ ] 开始修改推理逻辑

祝研究顺利！🎓


