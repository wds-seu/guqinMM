# inference_generate.py

import torch
import torch.nn.functional as F

# --- Sampling辅助函数 ---
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out

def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0

    indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(indices_to_remove, -float('Inf'))
    return logits

def sample_from_logits(logits, top_k=None, top_p=None, temperature=1.0):
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token

# --- 主函数 ---
def generate_music(
    model,
    score_tokens,
    style_label,
    device="cuda",
    method="sampling",  # 'sampling' or 'beam'
    max_length=1024,
    top_k=10,
    top_p=None,
    temperature=1.0,
    beam_width=5
):
    """
    根据score_tokens和style_label生成声学token序列
    """

    model.eval()
    states = None
    offset = 0

    score_tokens = score_tokens.to(device).unsqueeze(0)
    style_label = torch.tensor([style_label], device=device)

    if method == "sampling":
        input_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        generated_tokens = []

        for _ in range(max_length):
            logits, states, offset = model(
                score_tokens=score_tokens,
                style_labels=style_label,
                targets=input_token,
                states=states,
                offset=offset
            )
            next_logits = logits[:, -1, :]
            next_token = sample_from_logits(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
            generated_tokens.append(next_token.item())
            input_token = next_token.unsqueeze(1)

        return generated_tokens

    elif method == "beam":
        beams = [(torch.zeros(1, 1, dtype=torch.long, device=device), 0.0, states, offset)]

        for _ in range(max_length):
            new_beams = []
            for input_token, score, states, offset in beams:
                logits, states_new, offset_new = model(
                    score_tokens=score_tokens,
                    style_labels=style_label,
                    targets=input_token,
                    states=states,
                    offset=offset
                )
                logits = logits[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                for prob, idx in zip(topk_probs[0], topk_indices[0]):
                    new_seq = torch.cat([input_token, idx.view(1, 1)], dim=1)
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score, states_new, offset_new))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq = beams[0][0]
        return best_seq.squeeze(0).tolist()

    else:
        raise ValueError(f"Unsupported generation method: {method}")
