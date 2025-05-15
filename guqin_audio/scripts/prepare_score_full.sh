#!/bin/bash
set -e

echo "🚀 Step 1: 将总谱文文本按逗号分词为token序列..."

python3 scripts/text_to_tokenized_list.py \
  --text_file data/raw/full_text.txt \
  --output_path data/score_tokenized_raw.pt

echo "✅ 完成逗号分词并保存至 data/score_tokenized_raw.pt"

# --------------------------------------

echo "🚀 Step 2: 将小片段tokens均分组合为5首完整曲子..."

python3 scripts/prepare_score_tokenized.py \
  --input_path data/score_tokenized_raw.pt \
  --output_path data/score_tokenized.pt \
  --n_pieces 5

echo "✅ 完成曲子分组，生成标准谱文tokens data/score_tokenized.pt"

echo "🎯 完成 prepare_score_full 流程！"
