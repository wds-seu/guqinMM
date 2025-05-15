#!/bin/bash

# ========== 设置超参数 ==========
DATA_DIR="./data/finetune_data"       # 你的谱文+音频5首数据
CHECKPOINT_DIR="./checkpoints_finetune"
PRETRAINED_MODEL="./checkpoints/pretrain_latest.pt"   # 你的预训练好的模型路径
RESULT_DIR="./results"
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=1e-4
MAX_LENGTH=1024

# 设备设置（使用全部GPU）
export CUDA_VISIBLE_DEVICES=0,1

echo "🚀 Step 1: 开始微调 Finetune"
python train_finetune.py \
    --data_dir $DATA_DIR \
    --pretrained_model $PRETRAINED_MODEL \
    --save_dir $CHECKPOINT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH

# 取最新finetune后的checkpoint
LATEST_CKPT=$(ls -t $CHECKPOINT_DIR/*.pt | head -n 1)

echo "✅ Finetune完成，最新模型：$LATEST_CKPT"

# ========== 推理参数设置 ==========
METHOD="sampling"    # 可选 sampling / beam
TOP_K=20
TOP_P=0.9
TEMPERATURE=1.0
BEAM_WIDTH=5

SCORE_TOKENS_PATH="./data/inference/score_tokens.pt"  # 推理输入
STYLE_LABEL=0  # 流派类别，比如 exam = 0

mkdir -p $RESULT_DIR

echo "🚀 Step 2: 开始推理 Inference ($METHOD)"
python inference_generate_main.py \
    --model_ckpt $LATEST_CKPT \
    --score_tokens_path $SCORE_TOKENS_PATH \
    --style_label $STYLE_LABEL \
    --method $METHOD \
    --max_length $MAX_LENGTH \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    --beam_width $BEAM_WIDTH \
    --save_dir $RESULT_DIR

echo "🎵 生成完成，结果保存在 $RESULT_DIR 下！"

