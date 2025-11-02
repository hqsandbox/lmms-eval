#!/bin/bash
# Qwen2VL 本地实现安装脚本
# 在有 GPU 的机器上运行此脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "Qwen2VL 本地实现安装脚本"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}错误：请在 lmms-eval 根目录下运行此脚本${NC}"
    exit 1
fi

echo -e "${YELLOW}[1/5] 检查 Python 环境...${NC}"
python --version
if [ $? -ne 0 ]; then
    echo -e "${RED}错误：Python 未安装或不在 PATH 中${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 环境正常${NC}"
echo ""

echo -e "${YELLOW}[2/5] 安装 lmms-eval (editable 模式)...${NC}"
pip install -e .
if [ $? -ne 0 ]; then
    echo -e "${RED}错误：lmms-eval 安装失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ lmms-eval 安装成功${NC}"
echo ""

echo -e "${YELLOW}[3/5] 安装必要依赖...${NC}"
pip install transformers torch accelerate qwen-vl-utils decord pillow numpy requests
if [ $? -ne 0 ]; then
    echo -e "${RED}错误：依赖安装失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 依赖安装成功${NC}"
echo ""

echo -e "${YELLOW}[4/5] 验证本地实现...${NC}"
echo "运行测试脚本..."
python -c "
import sys
try:
    from lmms_eval.models.local_models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from lmms_eval.models.local_models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    print('✓ 本地 Qwen2VL 实现导入成功')
    sys.exit(0)
except Exception as e:
    print(f'✗ 导入失败: {e}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo -e "${RED}错误：本地实现验证失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 本地实现验证通过${NC}"
echo ""

echo -e "${YELLOW}[5/5] 检查 GPU 可用性...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU 可用：{torch.cuda.get_device_name(0)}')
    print(f'  GPU 数量: {torch.cuda.device_count()}')
    print(f'  CUDA 版本: {torch.version.cuda}')
else:
    print('⚠️  警告：未检测到 GPU，将使用 CPU（速度会很慢）')
"
echo ""

echo "========================================"
echo -e "${GREEN}安装完成！${NC}"
echo "========================================"
echo ""
echo "下一步："
echo ""
echo "1. 运行快速测试（可选）："
echo "   ${YELLOW}python test_local_qwen2vl.py${NC}"
echo ""
echo "2. 运行完整评测验证："
echo "   ${YELLOW}lmms-eval --model qwen2_vl --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct --tasks mme --batch_size 1${NC}"
echo ""
echo "3. 查看详细文档："
echo "   ${YELLOW}cat QUICK_START_LOCAL_QWEN2VL.md${NC}"
echo ""
echo "期望看到的验证信息："
echo "   ${GREEN}🔥 Using LOCAL Qwen2VL implementation from lmms_eval/models/local_models/qwen2_vl/ 🔥${NC}"
echo ""
echo "========================================"


