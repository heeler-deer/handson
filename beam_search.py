import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 定义字符表和编码器
chars = list("abcdefghijklmnopqrstuvwxyz ")
encoder = LabelEncoder()
encoder.fit(chars)
vocab_size = len(chars)

# 伪造的概率模型（用于模拟生成下一个字符的概率分布）
def mock_model_predict(prev_token_ids):
    """模拟给定前一个token，返回下一个token的概率分布"""
    last_token_id = prev_token_ids[-1]
    probs = np.full(vocab_size, 1.0 / vocab_size)  # 均匀分布
    if chars[last_token_id] == 'h':
        probs[encoder.transform(['e'])[0]] += 0.2
    elif chars[last_token_id] == 'e':
        probs[encoder.transform(['l'])[0]] += 0.2
    elif chars[last_token_id] == 'l':
        probs[encoder.transform(['l'])[0]] += 0.2
    elif chars[last_token_id] == 'o':
        probs[encoder.transform([' '])[0]] += 0.2
    probs /= probs.sum()
    return torch.tensor(probs)

# Beam Search 实现
def beam_search(start_token, beam_width=3, max_len=5):
    start_id = encoder.transform([start_token])[0]
    sequences = [([start_id], 0.0)]  # 每个元素是(token_id_list, log_prob)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            probs = mock_model_predict(seq)
            topk_probs, topk_ids = torch.topk(probs, beam_width)
            for i in range(beam_width):
                candidate_seq = seq + [topk_ids[i].item()]
                candidate_score = score + torch.log(topk_probs[i] + 1e-10).item()
                all_candidates.append((candidate_seq, candidate_score))

        # 选出得分最高的 beam_width 个序列
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]

        # 打印每一步 beam 状态
        print(f"\nStep {_ + 1}:")
        for i, (seq, score) in enumerate(sequences):
            decoded = ''.join(encoder.inverse_transform(seq))
            print(f"Beam {i + 1}: '{decoded}' | Score: {score:.4f}")

    return sequences

# 运行 Beam Search
final_sequences = beam_search(start_token='h', beam_width=3, max_len=5)

# 打印最终结果
print("\nFinal Result:")
for seq, score in final_sequences:
    decoded = ''.join(encoder.inverse_transform(seq))
    print(f"Sequence: '{decoded}' | Score: {score:.4f}")
