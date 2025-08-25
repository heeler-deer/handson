import torch
import numpy as np

# 1. 固定字符顺序及其 id
chars = list("abcdefghijklmnopqrstuvwxyz ")
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for ch, i in stoi.items()}        # int    -> string
vocab_size = len(chars)

# 2. 概率模型
def mock_model_predict(prev_token_ids):
    last_token_id = prev_token_ids[-1]
    last_char = itos[last_token_id]

    # 均匀先验
    probs = np.full(vocab_size, 1.0 / vocab_size)
    # 按字符名触发规则
    if last_char == 'h':
        probs[stoi['e']] += 0.2
    elif last_char == 'e':
        probs[stoi['l']] += 0.2
    elif last_char == 'l':
        probs[stoi['l']] += 0.1
        probs[stoi['o']] += 0.2
    elif last_char == 'o':
        probs[stoi[' ']] += 0.2
    probs /= probs.sum()
    return torch.tensor(probs, dtype=torch.float32)

# 3. Beam Search
def beam_search(start_token, beam_width=3, max_len=5):
    start_id = stoi[start_token]
    sequences = [([start_id], 0.0)]  # (token_id_list, log_prob)

    for step in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            probs = mock_model_predict(seq)
            topk_probs, topk_ids = torch.topk(probs, beam_width)
            for p, idx in zip(topk_probs, topk_ids):
                candidate_seq = seq + [idx.item()]
                candidate_score = score + torch.log(p + 1e-10).item()
                all_candidates.append((candidate_seq, candidate_score))

        # 保留得分最高的 beam_width 条
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]

        # 打印当前 beam
        print(f"\nStep {step+1}:")
        for i, (seq, score) in enumerate(sequences):
            decoded = ''.join(itos[tid] for tid in seq)
            print(f"Beam {i+1}: '{decoded}' | Score: {score:.4f}")

    return sequences

# 4. 运行
final_sequences = beam_search(start_token='h', beam_width=3, max_len=5)

print("\nFinal Result:")
for seq, score in final_sequences:
    decoded = ''.join(itos[tid] for tid in seq)
    print(f"Sequence: '{decoded}' | Score: {score:.4f}")