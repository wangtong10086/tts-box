import torch
from typing import List, Tuple
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FreePageQueue:
    def __init__(self, current_capacity: int):
        self.free_page_queue = deque(range(current_capacity))
    
    def dequeue_left(self) -> int:
        return self.free_page_queue.popleft() if self.free_page_queue else -1
    
    def enqueue_right(self, physical_page_id: int):
        self.free_page_queue.append(physical_page_id)
    
    def batch_enqueue_right(self, physical_page_id_list: List[int]):
        self.free_page_queue.extend(physical_page_id_list)


class PageTableManager:
    def __init__(self, block_size, num_head, headsize, initial_pages, max_pages, dtype, initial_expansion_step=10):
        self.block_size = block_size
        self.num_head = num_head
        self.headsize = headsize
        self.max_pages = max_pages
        self.expansion_step = initial_expansion_step  # 初始扩展步长
        self.current_capacity = initial_pages
        self.dtype = dtype
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 页面池
        self.page_pool = []
        for i in range(self.current_capacity):
            self.page_pool.append(torch.zeros((self.block_size, self.num_head, self.headsize), device=self.device, dtype=self.dtype))
        # 物理页的引用计数，初始时为0
        self.page_usage = [0] * self.current_capacity
        # 虚拟页--物理也映射表
        self.page_map = {}
        # 未完全填充的逻辑页编号列表
        self.remain_pages = []
        # 序列中每个token对应的物理页, eg, sequence_page_table[0]代表第一个token对应的物理页
        self.sequence_page_table = []
        
        # 空闲 page 队列  -- 左侧出队，右侧入队
        self.free_page_queue = FreePageQueue(self.current_capacity)

    def _expand_pool(self):
        if self.current_capacity >= self.max_pages:
            logging.info("Page pool has reached maximum capacity.")
            return

        # 混合策略扩展步长
        expansion_step = self.expansion_step if self.current_capacity <= 256 else min(16, self.max_pages - self.current_capacity)
        new_capacity = min(self.current_capacity + expansion_step, self.max_pages)
        additional_pages = new_capacity - self.current_capacity

        # 扩展页面池和页面使用跟踪数组
        for i in range(additional_pages):
            self.page_pool.append(torch.zeros((self.block_size, self.num_head, self.headsize), device=self.device, dtype=self.dtype))
            
        self.page_usage.extend([0] * additional_pages)
        self.free_page_queue.batch_enqueue_right(range(self.current_capacity, new_capacity))
        self.current_capacity = new_capacity

        if self.current_capacity <= 256:
            self.expansion_step *= 2

        logging.info(f"Expanded page pool to new capacity: {self.current_capacity}, expansion_step: {expansion_step}")


    def allocate_page(self, page_id: str) -> int:
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

        assert self.page_usage[page_idx] == 0, "Page usage count should be 0 before allocation."
        self.page_usage[page_idx] += 1
        self.page_map[page_id] = page_idx
        return page_idx
    
    def hash_block(self, block_token_idx: List[str]) -> str:
        return '#'.join(block_token_idx)

    def replace_first_masked_position(self, logical_id: str, token_id: int) -> Tuple[str, int]:
        parts = logical_id.split('#')
        first_masked_pos = parts.index('M')
        parts[first_masked_pos] = str(token_id)
        return '#'.join(parts), first_masked_pos
    
    def get_cached_pages(self):
        return self.page_pool
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
        
        assert b == 1, "Only support batch size of 1 for prefill"
        
        data = data.reshape(b*seq_len, num_head, head_size)
        
        # 划分 token_idx_list 为多个小块
        blocks = [list(map(str, token_idx_list[i:i + self.block_size])) for i in range(0, len(token_idx_list), self.block_size)]

        # 检查最后一个块的大小，如果小于 block_size，则进行填充
        if len(blocks[-1]) < self.block_size:
            blocks[-1].extend(['M'] * (self.block_size - len(blocks[-1])))

        # 为每个块分配页面并填充
        for idx, block in enumerate(blocks):
            logical_page_id = self.hash_block(block)
            mask_count = logical_page_id.count('M')
            if 'M' in logical_page_id:
                self.remain_pages.append(logical_page_id)
            physical_page_idx = self.allocate_page(logical_page_id)
            if physical_page_idx == -1:
                raise ValueError("Allocate memory out of max page size, Failed to allocate page for block:", block)
            fill_token_size = self.block_size - mask_count
            self.sequence_page_table.extend([physical_page_idx] * fill_token_size)
            # 对于每个块中的每个 token_idx，从 data 中提取对应的 token embedding
            start_pos = idx * self.block_size
            end_pos = min(start_pos + self.block_size, data.size(0))
            self.page_pool[physical_page_idx][0:end_pos-start_pos, :, :] = data[start_pos:end_pos, :, :]
            
    
    def decode(self, block_token_idx:List[int], data:torch.Tensor=None):
        assert data is not None, "data must be provided for decode"
        current_token_id = block_token_idx[-1]
        # 如果有未填满的page
        if self.remain_pages:
            # 找到未填满的page, 现在只考虑单个sequence，所以remain_pages只会有一个元素
            # 并且该元素一定指向先前生成token的尾端
            logical_remain_page_id = self.remain_pages[0]
            # 替换其中首个'M'字段，组成新的逻辑页id, 并查找第一个 'M' 在列表中的位置
            new_page_id, first_masked_pos = self.replace_first_masked_position(logical_remain_page_id, current_token_id)
            # 找到当前物理页id
            physical_current_page_id = self.page_map[logical_remain_page_id]
            # 是否已经有对应的page -- 如果有, 直接复用
            if new_page_id in self.page_map:
                # 找到更新的物理页
                physical_update_page_id = self.page_map[new_page_id]
                # 当前引用更新，不再引用旧的物理页
                self.page_usage[physical_current_page_id] -= 1
                self.page_usage[physical_update_page_id] += 1

                # 找到先前生成token有多少个mask，只考虑单个sequence，走入该分支，一定是1
                mask_count = logical_remain_page_id.count('M')
                # 对应可以找到该page填充的token数目
                fill_token_size = self.block_size - mask_count
                # 改写先前对应token的映射
                self.sequence_page_table[-fill_token_size:] = [physical_update_page_id] * fill_token_size
                # 将当前生成token对应的物理页添加到page table中
                self.sequence_page_table.append(physical_update_page_id)

                # 检查先前的物理页是否有别的引用,如果没有
                if self.page_usage[physical_current_page_id] == 0:
                   # 将先前的物理页放入空闲物理页队列中
                   self.free_page_queue.enqueue_right(physical_current_page_id)
                   # 删除掉对应的逻辑页--物理页映射
                   del self.page_map[logical_remain_page_id]
                   del self.remain_pages[0]
            else: # 如果没有, 往未填满的page中插入
                # 将数据插入page pool对应的位置当中
                self.page_pool[self.page_map[logical_remain_page_id]][first_masked_pos, :, :] = data
                # 删除旧的逻辑页面id -- 物理页id映射，因为逻辑页面id要更换
                del self.page_map[logical_remain_page_id]
                # 更新逻辑页id的映射
                self.page_map[new_page_id] = physical_current_page_id
                self.sequence_page_table.append(physical_current_page_id)
                # 说明当前逻辑页已被填满
                if new_page_id.count('M') == 0:
                    del self.remain_pages[0]
        else: # 如果没有，表示所有的物理页均已填满，要重新申请一个物理页
            # 设置新申请物理页的逻辑页id
            current_block_id = [str(current_token_id)] + ['M'] * (self.block_size - 1)
            logical_page_id = self.hash_block(current_block_id)

            # 将对应的逻辑页追加到remain_pages中去
            if 'M' in logical_page_id:
                self.remain_pages.append(logical_page_id)
            # 申请物理页
            physical_page_idx = self.allocate_page(logical_page_id)
            if physical_page_idx == -1:
                raise ValueError("Allocate memory out of max page size, Failed to allocate page for block:", physical_page_idx)
            # 将数据插入page pool对应的位置当中
            self.page_pool[physical_page_idx][0, :, :] = data 
            self.sequence_page_table.append(physical_page_idx)
            

# Helper function to initialize manager
def init_manager(block_size=2, num_head=2, headsize=2, initial_pages=2, max_pages=4, dtype=torch.float32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return PageTableManager(block_size=block_size, num_head=num_head, headsize=headsize,
                            initial_pages=initial_pages, max_pages=max_pages, dtype=dtype), device

def test_free_page_queue():
    # Initialize queue with a capacity of 5
    queue = FreePageQueue(5)
    assert queue.dequeue_left() == 0, "FreePageQueue: Expected first element to be 0"
    assert queue.dequeue_left() == 1, "FreePageQueue: Expected second element to be 1"
    queue.enqueue_right(5)
    assert queue.dequeue_left() == 2, "FreePageQueue: Expected third element to be 2"
    assert queue.dequeue_left() == 3, "FreePageQueue: Expected fourth element to be 3"
    print("FreePageQueue tests passed.")

def test_allocate_page():
    manager, _ = init_manager(initial_pages=5, max_pages=10)
    page_id = "page1"
    allocated_page = manager.allocate_page(page_id)
    assert allocated_page != -1, "Page allocation failed: No valid page allocated"
    assert manager.page_map[page_id] == allocated_page, "Page allocation failed: Incorrect page mapping"
    print("allocate_page test passed.")

def test_prefill():
    manager, device = init_manager()
    token_idx_list = [1, 2, 3, 4, 5, 6]
    data = torch.ones((1, len(token_idx_list), 2, 2), device=device, dtype=torch.float32)
    manager.prefill(token_idx_list, data=data)
    
    for page_id, idx in manager.page_map.items():
        assert (manager.page_pool[idx] == 1).all(), f"Prefill failed: Data mismatch on page {page_id}"
    print("Prefill test passed.")

def test_decode_fill_page():
    torch.manual_seed(1999)
    manager, device = init_manager()
    token_idx_list = [1, 2, 3, 4, 5]
    data = torch.ones((1, len(token_idx_list), 2, 2), device=device, dtype=torch.float32)
    manager.prefill(token_idx_list, data=data)
    
    print(manager.sequence_page_table)
    
    new_token_idx = [1, 2, 3, 4, 5, 6]
    token_data = torch.randn((2, 2), device=device, dtype=torch.float32)
    manager.decode(new_token_idx, data=token_data)

    print(manager.sequence_page_table)
    
    logical_page_id, _ = manager.replace_first_masked_position("5#M", 6)
    physical_page_idx = manager.page_map.get(logical_page_id, -1)
    
    assert physical_page_idx != -1, "Decode failed: No allocated page for new token."
    assert (manager.page_pool[physical_page_idx][1] == token_data).all(), "Decode failed: Data mismatch on decoded page"
    print("Decode fill reamin page test passed.")

def test_decode_reuse():
    torch.manual_seed(1999)
    manager, device = init_manager()
    
    vocab_size, embedding_dim = 10, 4
    embeddings = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.float32)
    token_idx_list = [1, 2, 3, 4, 3]
    data = embeddings[token_idx_list].view(1, len(token_idx_list), 2, 2)
    manager.prefill(token_idx_list, data=data)
    print(manager.sequence_page_table)
    
    new_token_id = 4
    token_idx_list.append(new_token_id)
    token_data = embeddings[new_token_id].view(1, 1, 2, 2)
    manager.decode(token_idx_list, data=token_data)
    print(manager.sequence_page_table)
    
    logical_page_id, _ = manager.replace_first_masked_position("3#M", 4)
    physical_page_idx = manager.page_map.get(logical_page_id, -1)
    
    assert physical_page_idx != -1, "Decode reuse failed: No valid page allocated."
    assert 2 in manager.free_page_queue.free_page_queue, "Decode reuse failed: Expected page not reused."
    assert (manager.page_pool[physical_page_idx][1] == token_data).all(), "Decode failed: Data mismatch on decoded page"
    print("Decode reuse test passed.")


def test_decode_new_page():
    torch.manual_seed(1999)
    manager, device = init_manager()
    
    vocab_size, embedding_dim = 10, 4
    embeddings = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.float32)
    token_idx_list = [1, 2, 3, 4]
    data = embeddings[token_idx_list].view(1, len(token_idx_list), 2, 2)
    manager.prefill(token_idx_list, data=data)
    print(manager.sequence_page_table)
    
    new_token_id = 5
    token_idx_list.append(new_token_id)
    token_data = embeddings[new_token_id].view(1, 1, 2, 2)
    manager.decode(token_idx_list, data=token_data)
    print(manager.sequence_page_table)

    logical_page_id, _ = manager.replace_first_masked_position("M#M", 5)
    physical_page_idx = manager.page_map.get(logical_page_id, -1)
    
    assert physical_page_idx != -1, "Decode failed: No valid page allocated."
    assert (manager.page_pool[physical_page_idx][0] == token_data).all(), "Decode failed: Data mismatch on decoded page"
    print("Decode new page test passed.")


test_free_page_queue()
test_allocate_page()
test_prefill()
test_decode_fill_page()
test_decode_reuse()
test_decode_new_page()