import torch

# 假设的维度
batch = 2
seq_len = 3
heads = 4
d_k = 5
emb_size = heads * d_k

# 随机生成数据
q = torch.randn(batch, seq_len, heads, d_k)
self_pos_bias_v = torch.randn(heads, d_k)
pos_emb = torch.randn(batch, seq_len, emb_size)
self_linear_pos = torch.nn.Linear(emb_size, heads * d_k)

# 原始代码
q_with_bias_v_original = (q + self_pos_bias_v).transpose(1, 2)
p_original = self_linear_pos(pos_emb).view(batch, -1, heads, d_k).transpose(1, 2)
matrix_bd_original = torch.matmul(q_with_bias_v_original, p_original.transpose(-2, -1))

# 优化后的代码
q = q.transpose(1, 2).contiguous()
q_with_bias_v_optimized = q + self_pos_bias_v
p_optimized = self_linear_pos(pos_emb).view(batch, heads, seq_len, d_k).contiguous()
matrix_bd_optimized = torch.einsum('b h s d, b h d q -> b h s q', q_with_bias_v_optimized, p_optimized)

# 验证结果是否一致
print(torch.allclose(matrix_bd_original, matrix_bd_optimized, atol=1e-6))