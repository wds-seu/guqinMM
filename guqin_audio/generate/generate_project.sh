#!/bin/bash

echo "Creating Guqin Generation Project Structure..."

# 创建文件夹
mkdir -p data/raw
mkdir -p data/raw_75h
mkdir -p data/processed_wav
mkdir -p data/processed_75h
mkdir -p data/processed
mkdir -p model
mkdir -p utils
mkdir -p scripts
mkdir -p checkpoints
mkdir -p outputs

# 创建空白文件
touch requirements.txt
touch README.md
touch run_all.sh
touch train_and_generate_guqin.py

# 脚本文件
touch scripts/prepare_wav.py
touch scripts/build_vocab_and_tokenize.py
touch scripts/encodec_extract_tokens.py
touch scripts/semantic_tokenize.py
touch scripts/assemble_dataset.py
touch scripts/prepare_75h_dataset.py
touch scripts/data_prepare.sh
touch scripts/prepare_75h_data.sh

# 模型文件
touch model/semantic_token_generator.py
touch model/acoustic_token_generator.py

# 工具类
touch utils/trainer.py
touch utils/inference.py
touch utils/audio_decoder.py

echo "✅ Project structure created successfully!"
