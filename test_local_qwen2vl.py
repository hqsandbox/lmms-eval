#!/usr/bin/env python3
"""
快速测试脚本：验证本地 Qwen2VL 实现是否正常工作

使用方法：
    python test_local_qwen2vl.py

预期输出：
    - 应该看到 "🔥 Using LOCAL Qwen2VL implementation..." 的打印
    - 模型能够成功加载并生成输出
"""

import torch
from PIL import Image
import requests
from io import BytesIO

# 导入本地实现
print("=" * 80)
print("测试本地 Qwen2VL 实现")
print("=" * 80)

print("\n[1/4] 导入本地 Qwen2VL 实现...")
try:
    from lmms_eval.models.local_models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from lmms_eval.models.local_models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    print("✓ 成功导入本地实现")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    print("\n请确保已运行: pip install -e .")
    exit(1)

print("\n[2/4] 加载模型和处理器...")
model_name = "Qwen/Qwen2-VL-7B-Instruct"
try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    print(f"✓ 成功加载模型: {model_name}")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    exit(1)

print("\n[3/4] 准备测试图像和文本...")
try:
    # 使用一个简单的测试图像
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    print("✓ 成功加载测试图像")
    
    # 准备消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is in this image?"}
            ]
        }
    ]
    print("✓ 准备好测试消息")
except Exception as e:
    print(f"✗ 准备失败: {e}")
    exit(1)

print("\n[4/4] 运行推理...")
print("-" * 80)
print("⚠️  注意：你应该会看到下面的验证信息：")
print("   '🔥 Using LOCAL Qwen2VL implementation from lmms_eval/models/local_models/qwen2_vl/ 🔥'")
print("-" * 80)

try:
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 注意：这里需要使用 qwen_vl_utils.process_vision_info
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        print("\n⚠️  警告：qwen_vl_utils 未安装，使用简化处理")
        print("   请运行: pip install qwen-vl-utils")
        image_inputs = [image]
        video_inputs = None
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 生成输出（这里会触发 forward，应该能看到我们的 print）
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128
        )
    
    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    print("-" * 80)
    print("✓ 推理成功完成！")
    print(f"\n生成的输出: {output_text[0]}")
    
except Exception as e:
    print(f"✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print("\n总结：")
print("1. 如果你看到了 '🔥 Using LOCAL...' 的输出，说明本地实现已经生效")
print("2. 如果模型成功生成了输出，说明本地实现工作正常")
print("3. 现在你可以修改 modeling_qwen2_vl.py 来改变推理行为")
print("\n下一步：运行完整的 lmms-eval 评测，对比结果是否一致")
print("=" * 80)


