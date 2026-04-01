#!/bin/bash
# 固定高层模板前沿评估 - 完整流程

set -e

echo "========================================="
echo "固定高层模板前沿评估"
echo "========================================="

# 检查 checkpoint 是否存在
echo ""
echo "检查 checkpoint..."
for seed in 42 43 44 45 46; do
    high_model="outputs/formal_multiseed/hierarchical/formal_final_seed${seed}/stage3_high_refine/high_model.zip"
    low_model="outputs/formal_multiseed/hierarchical/formal_final_seed${seed}/stage3_high_refine/low_model.zip"
    
    if [ ! -f "$high_model" ]; then
        echo "警告: 缺少 high model: $high_model"
    fi
    if [ ! -f "$low_model" ]; then
        echo "警告: 缺少 low model: $low_model"
    fi
done

# 选择评估模式
MODE="${1:-smoke}"

if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
    echo "错误: 模式必须是 'smoke' 或 'full'"
    echo "用法: $0 [smoke|full]"
    exit 1
fi

echo ""
echo "评估模式: $MODE"
if [ "$MODE" = "smoke" ]; then
    echo "  - N_eval=20 (快速验证)"
    echo "  - 预计时间: 30 分钟"
else
    echo "  - N_eval=64 (正式评估)"
    echo "  - 预计时间: 2-3 小时"
fi

# 运行评估
echo ""
echo "========================================="
echo "步骤 1: 运行评估"
echo "========================================="
python scripts/eval_fixed_high_frontier.py --mode "$MODE"

# 分析结果
echo ""
echo "========================================="
echo "步骤 2: 分析结果"
echo "========================================="
python analysis/fixed_high_frontier/analyze_results.py \
    --data-dir analysis/fixed_high_frontier \
    --output-dir analysis/fixed_high_frontier/plots

# 显示结果
echo ""
echo "========================================="
echo "评估完成"
echo "========================================="
echo ""
echo "结果文件:"
echo "  - 数据: analysis/fixed_high_frontier/*.csv"
echo "  - 图表: analysis/fixed_high_frontier/plots/*.png"
echo "  - 报告: analysis/fixed_high_frontier/plots/final_judgment.md"
echo ""
echo "查看最终判断:"
echo "  cat analysis/fixed_high_frontier/plots/final_judgment.md"
echo ""
