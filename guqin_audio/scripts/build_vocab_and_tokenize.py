import os
from collections import Counter
import torch

def build_vocab_and_tokenize(txt_file, save_vocab_path, save_tokenized_path):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    vocab_counter = Counter()
    tokenized_lines = []
    for line in lines:
        tokens = line.split(' ')  # 空格分词
        vocab_counter.update(tokens)
        tokenized_lines.append(tokens)

    vocab = {token: idx for idx, (token, _) in enumerate(vocab_counter.most_common())}
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)

    tokenized_ids = []
    for tokens in tokenized_lines:
        ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
        tokenized_ids.append(ids)

    torch.save(vocab, save_vocab_path)
    torch.save(tokenized_ids, save_tokenized_path)
    print(f"Vocab saved to {save_vocab_path}, tokenized data saved to {save_tokenized_path}")

if __name__ == "__main__":
    build_vocab_and_tokenize('./data/raw/full_text.txt', './data/vocab.pt', './data/score_tokenized.pt')
