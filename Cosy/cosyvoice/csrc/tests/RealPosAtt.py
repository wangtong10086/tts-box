import math
from typing import Tuple

import torch
from torch import nn
from page_manger import PageTableManager

import attention_cuda

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        #p_attn = attn
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                CosyVoice.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache



class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    
    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        
        # (batch, head, time1, 1)
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        # (batch, head, time1, 2*time1)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        
        # (batch, head, 2*time1, time1)
        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        
        
        # (batch, head, 2*time1+1, time1) --> (batch, head, 2*time1-1, time1) --> (batch, head, time1, 2*time1-1) --> (batch, head, time1, time1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]  # only keep the positions from 0 to time2
        return x
    
   
    

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        key_cache: torch.Tensor, 
        value_cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if key_cache.size(0) > 0:
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        print("k.shape: ", k.shape) # torch.Size([1, 16, 11, 64])
        print("v.shape: ", v.shape)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        print("p: ", p.shape)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)
        

        # (batch, head, time1, d_k)
        print("shape info")
        print("q: ", q.shape)
        print(self.pos_bias_u.shape)
        print("self.pos_bias_v: ", self.pos_bias_v.shape)
        print("self.pos_bias_u.shape: ", self.pos_bias_u.shape)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        print("q_with_bias_u.shape: ", q_with_bias_u.shape)  # [1, 16, 1, 64]
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        print("q_with_bias_v.shape: ", q_with_bias_v.shape)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        print("matrix_ac: ", matrix_ac.shape) # [1, 16, 1, 11]

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # do out of cuda kernel
        
        print("matrix_bd: ", matrix_bd.shape) # [1, 16, 1, 1]
        # NOTE(Xiang Lyu): Keep rel_shift since espnet rel_pos_emb is used
        if matrix_ac.shape != matrix_bd.shape:   # do out of cuda kernel
            matrix_bd = self.rel_shift(matrix_bd)

        print("matrix_bd: ", matrix_bd.shape)
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), k, v
    
    
    def infer(
        self,
        num_heads: int,
        head_size: int,
        query_index: int,
        kv_current_index: int,
        kv_indices: list,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        pos_emb: torch.Tensor = torch.empty(0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        block_size = 16
        key_cache = torch.squeeze(key_cache, dim=0).transpose(0, 1)   # [16, 10, 64]  -- > [10, 16, 64] 
        value_cache = torch.squeeze(value_cache, dim=0).transpose(0, 1)   # [16, 10, 64] -- > [10, 16, 64] 
        
        key_cache_manger = PageTableManager(block_size=block_size, num_head=num_heads, headsize=head_size,
                                       initial_pages=16, max_pages=512, dtype=query.dtype)
        value_cache_manger = PageTableManager(block_size=block_size, num_head=num_heads, headsize=head_size,
                                       initial_pages=16, max_pages=512, dtype=query.dtype)
        
        key_cache_manger.prefill(kv_indices, key_cache)
        value_cache_manger.prefill(kv_indices, value_cache)
        
    
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # [1, 1, 16, 64]
        q = torch.squeeze(q, dim=0) # [1, 16, 64]
        k = torch.squeeze(k, dim=0) # [1, 16, 64]
        v = torch.squeeze(v, dim=0) # [1, 16, 64]
        
        kv_indices.append(kv_current_index)
        key_cache_manger.decode(kv_indices, k)
        value_cache_manger.decode(kv_indices, v)
        
        key_cache = key_cache_manger.get_cached_pages()
        value_cache = value_cache_manger.get_cached_pages()
        
        qk_shape = (1, num_heads, key_cache.size(0)+1)
        
        q_with_bias_v = (q + self.pos_bias_v) #[1, 16, 64]
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = torch.squeeze(p.transpose(1, 2).transpose(-2, -1), dim=-1) # torch.Size([1, 21, 16, 64]) -- > torch.Size([1, 16, 64, 21]) --> torch.Size([16, 64, 21])
        matrix_bd = torch.matmul(q_with_bias_v, p) # [1, 16, 21]
        
        if qk_shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd) # [1, 16, 11]
        
        scale = float(1.0 / (head_size ** 0.5))
        
        context_lens = [key_cache.size(0)+1] # 生成一个长度为 num_tokens 的列表 context_lens，每个元素都是从 1 到 MAX_SEQ_LEN 之间的随机整数
        max_context_len = max(context_lens)
        context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')
        
        block_table = key_cache_manger.page_map
        
        block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
        
        output = torch.empty(1, num_heads, head_size, dtype=query.dtype, device='cuda')
        
        attention_cuda.xl_single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.pos_bias_u,
            matrix_bd,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
        )
        

def make_vocab(vocab_size, embedding_dim, device, dtype):
    embeddings = torch.randn(vocab_size, embedding_dim, device=device, dtype=dtype)
    return embeddings
      
        
    

def test_att():
    
    # 设置随机种子
    torch.manual_seed(1999)
    
    dtype = torch.float32
    
    embeddings = make_vocab(vocab_size=5000, embedding_dim=1024, device='cuda', dtype=dtype)

    # 定义 Multi-Head Attention 的参数
    dim = 1024     # dim
    dropout_rate = 0.0  # Dropout rate
    batch_size = 1  # Number of batches
    time1 = 1      # Length of the query
    time2 = 1      # Length of the query
    cache_t = 10
    head_size = 64
    head_num = dim // head_size

    # 创建 RelPositionMultiHeadedAttention 的实例
    attention_layer = RelPositionMultiHeadedAttention(n_head=head_num, n_feat=dim, dropout_rate=dropout_rate)

    # 从 embeddings 中获取 query
    query_indices = torch.randint(0, 5000, (batch_size, time1)).to('cuda')  # 随机生成查询词汇索引
    query = embeddings[query_indices].to('cuda').view(batch_size, time1, dim)  # 获取 query 嵌入

    # 从 embeddings 中获取 key_cache 和 value_cache
    vocab_indices = torch.randint(0, 5000, (1, head_num, cache_t)).to('cuda')  # 随机生成词汇索引
    key_embeddings = embeddings[vocab_indices].to('cuda')  # 获取 key 嵌入
    value_embeddings = embeddings[vocab_indices].to('cuda')  # 获取 value 嵌入     

    # 调整形状以匹配 key_cache 和 value_cache
    key_cache = key_embeddings.view(1, head_num, cache_t, head_size)
    value_cache = value_embeddings.view(1, head_num, cache_t, head_size)

    # 创建其他必要的张量
    mask = torch.ones(batch_size, 1, time2).to('cuda')     # Mask tensor (all ones)
    pos_emb = torch.rand(batch_size, 2*cache_t+1, dim).to('cuda')  # Positional embedding tensor

    # 调用注意力层的 forward 方法
    output, key_cache, value_cache = attention_layer(query, query, query, mask, pos_emb, key_cache, value_cache)

    # 打印输出和新缓存的形状
    print("Output shape:", output.shape)           # 预期形状: (batch_size, time1, n_feat)
    print("New cache shape:", key_cache.shape)      # 预期形状: (1, n_head, time2 + 0, n_feat * 2)
    
    print(output)
    

def test_shif():
    torch.manual_seed(1999)
    query = torch.rand(1, 2, 4, 3)
    print(query)
    dim = 1024     # dim
    dropout_rate = 0.0  # Dropout rate
    head_size = 64
    head_num = dim // head_size
    attention_layer = RelPositionMultiHeadedAttention(n_head=head_num, n_feat=dim, dropout_rate=dropout_rate)
    x = attention_layer.rel_shift(query)
    print(x)



if __name__ == '__main__':
    
    test_att()
    #test_shif()
    #test_mask()
    