import torch
import hashlib

class PageTableManager:
    def __init__(self, block_size, num_head, headsize, initial_pages, max_pages, initial_expansion_step=10):
        self.block_size = block_size
        self.num_head = num_head
        self.headsize = headsize
        self.max_pages = max_pages
        self.expansion_step = initial_expansion_step  # 初始扩展步长
        self.current_capacity = initial_pages

        # 页面池和页面跟踪
        self.page_pool = torch.empty((self.current_capacity, block_size, num_head, headsize), device='cuda')
        self.page_usage = [False] * self.current_capacity
        self.page_map = {}
        self.remain_block = {}

    def _expand_pool(self):
        if self.current_capacity >= self.max_pages:
            print("Page pool has reached maximum capacity.")
            return

        # 混合策略扩展步长
        if self.current_capacity < 50:
            expansion_step = self.expansion_step
        elif self.current_capacity < 200:
            expansion_step = self.expansion_step * 2
        else:
            remaining_capacity = self.max_pages - self.current_capacity
            expansion_step = min(remaining_capacity, max(self.expansion_step, 10))

        # 计算新容量，并确保不超过 max_pages
        new_capacity = min(self.current_capacity + expansion_step, self.max_pages)
        additional_pages = new_capacity - self.current_capacity

        # 扩展页面池和页面使用跟踪数组
        self.page_pool = torch.cat(
            [self.page_pool, torch.empty((additional_pages, self.block_size, self.num_head, self.headsize), device='cuda')],
            dim=0
        )
        self.page_usage.extend([False] * additional_pages)
        self.current_capacity = new_capacity

        # 更新扩展步长
        if self.current_capacity < 200:
            self.expansion_step *= 2
        else:
            self.expansion_step += 5

        print(f"Expanded page pool to new capacity: {self.current_capacity}, expansion_step: {expansion_step}")

    def allocate_page(self, page_id):
        if page_id in self.page_map:
            return self.page_map[page_id]

        for page_idx in range(self.current_capacity):
            if not self.page_usage[page_idx]:
                self.page_usage[page_idx] = True
                self.page_map[page_id] = page_idx
                return page_idx

        if self.current_capacity < self.max_pages:
            self._expand_pool()
            return self.allocate_page(page_id)

        print("Error: No available pages.")
        return -1

    def release_page(self, page_id):
        if page_id in self.page_map:
            page_idx = self.page_map[page_id]
            self.page_usage[page_idx] = False
            del self.page_map[page_id]
            
    
    def hash_block(self, block_token_idx: list):
        # 使用 hashlib 的 sha256 来获得更加稳定的哈希值
        block_token_tuple = tuple(block_token_idx)
        return hashlib.sha256(str(block_token_tuple).encode('utf-8')).hexdigest()
            
    def prefill(self, token_idx_list: list, data: torch.Tensor = None):
        """
        预填充页面池，根据 token_idx_list 划分块并将数据填充到页面中。

        Args:
            token_idx_list (list): 要处理的 token 索引列表。
            data (Tensor): 形状为 [b, seq_len, num_head, head_size] 的数据张量，用于填充页面池中的相应块。
        """
        # 划分 token_idx_list 为多个小块
        blocks = [token_idx_list[i:i + self.block_size] for i in range(0, len(token_idx_list), self.block_size)]

        # 检查最后一个块的大小，如果小于 block_size，则进行填充
        if len(blocks[-1]) < self.block_size:
            remain_len = self.block_size - len(blocks[-1])
            blocks[-1] += [-1] * remain_len  # 使用 += 避免多次调用 extend
            page_id = self.hash_block(blocks[-1])
            self.remain_block[page_id] = remain_len

        # 为每个块分配页面并填充
        for idx, block in blocks:
            page_id = self.hash_block(block)
            allocated_page_idx = self.allocate_page(page_id)

             # 如果有数据传入，且处于 prefill 阶段，拷贝数据到页面
            if data is not None:
                # 对于每个块中的每个 token_idx，从 data 中提取对应的 token embedding
                current_block_tokens = idx * self.block_size
                data_block = data[current_block_tokens:current_block_tokens + self.block_size]
                self.page_pool[self.page_map[page_id]] = data_block
                        


       
    
    def decode(self, block_token_idx:list, data:torch.Tensor=None):
        pass

    