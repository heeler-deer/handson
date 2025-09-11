import torch, math

def flash_attn(q, k, v, causal=False, Br=64, Bc=64):
    """
    q,k,v: (B, H, N, d)
    causal: bool
    return: (B, H, N, d)
    """
    B, H, N, d = q.shape
    device = q.device
    
    # 缩放因子
    scale = 1.0 / math.sqrt(d)
    
    o = torch.zeros_like(q)

    # 简单起见，仍然使用循环。向量化处理可以提高效率但会降低可读性。
    for b in range(B):
        for h in range(H):
            Q, K, V = q[b, h], k[b, h], v[b, h]  # (N, d)
            
            # 初始化全局状态
            Out = torch.zeros_like(Q)
            row_max = torch.full((N,), -torch.inf, device=device, dtype=torch.float32)
            row_sum = torch.zeros(N, device=device, dtype=torch.float32)

            # 外层循环: 遍历查询块 (rows)
            for i in range(0, N, Br):
                ir = slice(i, i + Br)
                Qi = Q[ir]  # (Br, d)
                
                # 加载当前查询块的局部状态
                mi = row_max[ir]  # (Br,)
                li = row_sum[ir]  # (Br,)
                Oi = Out[ir]      # (Br, d)

                # 内层循环: 遍历键/值块 (columns)
                for j in range(0, N, Bc):
                    jc = slice(j, j + Bc)
                    
                    # 对于因果掩码，如果整个 K 块都在 Q 块的未来，则跳过
                    if causal and j > i + Br - 1:
                        continue
                    
                    Kj, Vj = K[jc], V[jc] # (Bc, d)
                    
                    # 核心计算
                    Sij = (Qi @ Kj.T) * scale # (Br, Bc)

                    # 应用因果掩码 (如果需要)
                    if causal:
                        # 创建一个掩码，当 key_idx > query_idx 时为 True
                        query_indices = torch.arange(i, i + Br, device=device).view(-1, 1)
                        key_indices = torch.arange(j, j + Bc, device=device).view(1, -1)
                        mask = query_indices < key_indices
                        Sij.masked_fill_(mask, -torch.inf)
                    
                    # --- 在线 Softmax 更新 ---
                    # 1. 计算新的最大值
                    m_ij_new = torch.max(Sij, dim=1).values # (Br,)
                    m_new = torch.maximum(mi, m_ij_new)
                    
                    # 2. 计算 P_ij = exp(S_ij - m_new)
                    p = torch.exp(Sij - m_new[:, None]) # (Br, Bc)
                    
                    # 3. 更新分母 l
                    li_scaled = li * torch.exp(mi - m_new)
                    l_new = li_scaled + torch.sum(p, dim=1)
                    
                    # 4. 更新输出 O  <-- 这是最关键的修正！
                    Oi_scaled = Oi * li_scaled[:, None]
                    Oi_new_block = p @ Vj # (Br, d)
                    Oi = (Oi_scaled + Oi_new_block) / l_new[:, None]
                    
                    # 更新用于下一次 j 循环的局部状态
                    mi, li = m_new, l_new
                    
                # 【修正2】: 内层循环结束后，将最终结果写回全局状态
                Out[ir] = Oi
                row_max[ir] = mi
                row_sum[ir] = li

            o[b, h] = Out
            
    return o

if __name__ == "__main__":
    B,H,N,d = 2,4,128,32
    q = torch.randn(B,H,N,d) / math.sqrt(d)
    k = torch.randn(B,H,N,d) / math.sqrt(d)
    v = torch.randn(B,H,N,d)
    with torch.no_grad():
        mask = torch.triu(torch.full((N,N), -float('inf')), diagonal=1)
        ref = torch.softmax(q@k.transpose(-2,-1) + mask, -1) @ v
    out = flash_attn(q, k, v, causal=True)
    print("max diff:", (out-ref).abs().max().item())