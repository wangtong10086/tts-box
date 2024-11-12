import torch
from typing import List
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class FreePageQueue():
    def __init__(self, current_capacity):
        self.free_page_queue = deque(range(current_capacity))
        
    def dequeue_left(self):
        if self.free_page_queue:
            return self.free_page_queue.popleft()
        else:
            return -1
    def enqueue_right(self, item):
        self.free_page_queue.append(item)
    
    def batch_enqueue_right(self, items: list):
        self.free_page_queue.extend(items)



class PageTableManager:
    def __init__(self, block_size, num_head, headsize, initial_pages, max_pages, dtype, initial_expansion_step=10):
        self.block_size = block_size
        self.num_head = num_head
        self.headsize = headsize
        self.max_pages = max_pages
        self.expansion_step = initial_expansion_step  # 初始扩展步长
        self.current_capacity = initial_pages
        self.dtype = dtype

        # 页面池和页面跟踪
        self.page_pool = []
        for i in range(self.current_capacity):
            self.page_pool.append(torch.zeros((self.block_size, self.num_head, self.headsize), device='cuda', dtype=self.dtype))
        self.page_usage = [0] * self.current_capacity
        self.page_map = {}
        self.remain_block = {}
        
        # 空闲 page 队列  -- 左侧出队，右侧入队
        self.free_page_queue = FreePageQueue(self.current_capacity)

    def _expand_pool(self):
        if self.current_capacity >= self.max_pages:
            logging.info("Page pool has reached maximum capacity.")
            return

        # 混合策略扩展步长
        if self.current_capacity <= 256:
            expansion_step = self.expansion_step
        else:
            remaining_capacity = self.max_pages - self.current_capacity
            expansion_step = min(remaining_capacity, 16)

        # 计算新容量，并确保不超过 max_pages
        new_capacity = min(self.current_capacity + expansion_step, self.max_pages)
        additional_pages = new_capacity - self.current_capacity

        # 扩展页面池和页面使用跟踪数组
        for i in range(additional_pages):
            self.page_pool.append(torch.zeros((self.block_size, self.num_head, self.headsize), device='cuda', dtype=self.dtype))
            
        self.page_usage.extend([0] * additional_pages)
        self.current_capacity = new_capacity

        # 处理 free_page_queue 为空的情况
        if not self.free_page_queue:
            start_index = 0
        else:
            start_index = self.free_page_queue[-1] + 1

        new_pages = list(range(start_index, start_index + additional_pages))
        self.free_page_queue.batch_enqueue_right(new_pages)

        # 更新扩展步长
        if self.current_capacity <= 256:
            self.expansion_step *= 2

        logging.info(f"Expanded page pool to new capacity: {self.current_capacity}, expansion_step: {expansion_step}")

    def allocate_page(self, page_id):
        if page_id in self.page_map:
            self.page_usage[self.page_map[page_id]] += 1
            return self.page_map[page_id]

        page_idx = self.free_page_queue.dequeue_left()

        if page_idx == -1:
            if self.current_capacity >= self.max_pages:
                logging.error("Page pool has reached maximum capacity.")
                return -1
            self._expand_pool()
            return self.allocate_page(page_id)

        if self.page_usage[page_idx] == 0:
            self.page_usage[page_idx] += 1
            self.page_map[page_id] = page_idx
            return page_idx
        else:
            logging.error(f"Page {page_idx} is already in use. please check queue.")
            
    
    def hash_block(self, block_token_idx: List[str]):
        hash_id = ""
        for token_idx in block_token_idx:
            hash_id += token_idx
        return hash_id
            
    def prefill(self, token_idx_list: list, data: torch.Tensor = None):
        """
        预填充页面池，根据 token_idx_list 划分块并将数据填充到页面中。
        映射表 str(token_idx_list) -- > 物理块id
        Args:
            token_idx_list (list): 要处理的 token 索引列表。
            data (Tensor): 形状为 [b, seq_len, num_head, head_size] 的数据张量，用于填充页面池中的相应块。
        
        """
        assert data is not None, "data must be provided for prefill"
        
        b, seq_len, num_head, head_size = data.shape
        data = data.reshape(b*seq_len, num_head, head_size)
        
        # 划分 token_idx_list 为多个小块
        blocks = [str(token_idx_list[i:i + self.block_size]) for i in range(0, len(token_idx_list), self.block_size)]

        # 检查最后一个块的大小，如果小于 block_size，则进行填充
        if len(blocks[-1]) < self.block_size:
            remain_len = self.block_size - len(blocks[-1])
            blocks[-1] += ['M'] * remain_len  # 使用 += 避免多次调用 extend

        # 为每个块分配页面并填充
        for idx, block in blocks:
            page_id = self.hash_block(block)
            allocated_page_idx = self.allocate_page(page_id)
            if allocated_page_idx == -1:
                raise ValueError("Allocate memory out of max page size, Failed to allocate page for block:", block)
            
            # 对于每个块中的每个 token_idx，从 data 中提取对应的 token embedding
            start_pos = idx * self.block_size
            end_pos = min(start_pos + self.block_size, data.shape[0])
            data_block = data[start_pos:end_pos, :, :]
            self.page_pool[allocated_page_idx] = data_block
            
    
    def decode(self, block_token_idx:list, data:torch.Tensor=None):
        pass

    