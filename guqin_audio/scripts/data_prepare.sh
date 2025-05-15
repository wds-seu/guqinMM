#!/bin/bash

echo "Step 1: mp3 -> wav格式转换"
python scripts/prepare_wav.py

echo "Step 2: 建立谱文词表并分词编码"
python scripts/build_vocab_and_tokenize.py

echo "Step 3: 提取声学tokens (Encodec编码)"
python scripts/encodec_extract_tokens.py

echo "Step 4: 提取语义tokens (w2v-BERT + KMeans聚类)"
python scripts/semantic_tokenize.py

echo "Step 5: 组装成标准训练样本 (train/val/test)"
python scripts/assemble_dataset.py

echo "✅ 数据准备完成，所有文件已保存到 data/processed/ 目录下！"
