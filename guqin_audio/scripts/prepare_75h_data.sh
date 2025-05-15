#!/bin/bash

set -e

# ======================
# 配置
# ======================

RAW_75H_DIR="./data/raw_75h"              # 原始75h音频数据（不同流派文件夹）
PROCESSED_WAV_DIR="./data/processed_wav_75h"   # 重采样后的输出
ACOUSTIC_TOKEN_DIR="./data/acoustic_tokens_75h" # 声学token输出
OUTPUT_PT="./data/processed_75h/75h_dataset.pt" # 最终合并输出
SCRIPT_DIR="./scripts"                     # 脚本目录

# 流派标签映射（根据文件夹名字）
declare -A STYLE_MAP
STYLE_MAP=( ["exam"]=0 ["master"]=1 ["silkqin"]=2 ["other"]=3 )

# ======================
# 第一步：重采样
# ======================
echo "🔄 Step 1: Resampling audio to 24kHz mono..."

mkdir -p $PROCESSED_WAV_DIR

for style in "${!STYLE_MAP[@]}"
do
    mkdir -p "$PROCESSED_WAV_DIR/$style"
    for file in "$RAW_75H_DIR/$style"/*.mp3
    do
        filename=$(basename "$file" .mp3)
        ffmpeg -i "$file" -ar 24000 -ac 1 "$PROCESSED_WAV_DIR/$style/${filename}.wav" -y
    done
done

echo "✅ Resampling done."

# ======================
# 第二步：提取EnCodec声学token
# ======================
echo "🔄 Step 2: Extracting acoustic tokens with EnCodec..."

mkdir -p $ACOUSTIC_TOKEN_DIR

python3 $SCRIPT_DIR/extract_acoustic_tokens_75h.py \
    --input_dir $PROCESSED_WAV_DIR \
    --output_dir $ACOUSTIC_TOKEN_DIR \
    --style_map "${STYLE_MAP[@]}"

echo "✅ Acoustic tokens extraction done."

# ======================
# 第三步：整理成训练数据
# ======================
echo "🔄 Step 3: Assembling into single .pt dataset..."

python3 $SCRIPT_DIR/assemble_75h_dataset.py \
    --input_dir $ACOUSTIC_TOKEN_DIR \
    --output_path $OUTPUT_PT

echo "✅ 75h dataset preparation finished."

