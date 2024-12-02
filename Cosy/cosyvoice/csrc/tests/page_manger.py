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
    def __init__(self, block_size, num_head, headsize, initial_pages, max_pages, dtype, x_factor, num_layers, initial_expansion_step=10):
        self.block_size = block_size
        self.num_head = num_head
        self.headsize = headsize
        self.max_pages = max_pages
        self.expansion_step = initial_expansion_step  # 初始扩展步长
        self.current_capacity = initial_pages
        self.dtype = dtype
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.x_factor = x_factor
        self.num_layers = num_layers

        # 页面池，初始化存储 k_cache 和 v_cache 的数据
        self.k_cache_pool = {
            layer: torch.zeros((self.current_capacity, self.num_head, self.headsize // self.x_factor, self.block_size, self.x_factor),
                               device=self.device, dtype=self.dtype)
            for layer in range(self.num_layers)
        }
        self.v_cache_pool = {
            layer: torch.zeros((self.current_capacity, self.num_head, self.headsize, self.block_size),
                               device=self.device, dtype=self.dtype)
            for layer in range(self.num_layers)
        }
        
        # 物理页的引用计数，初始时为0
        self.page_usage = {layer: [0] * self.current_capacity for layer in range(self.num_layers)}
        # 虚拟页--物理页映射表
        self.page_map = {layer: {} for layer in range(self.num_layers)}
        # 未完全填充的逻辑页编号列表
        self.remain_pages = {layer: [] for layer in range(self.num_layers)}
        # 序列中每个 token 对应的物理页
        self.sequence_page_table = {layer: [] for layer in range(self.num_layers)}
        
        # 空闲 page 队列  -- 左侧出队，右侧入队
        self.free_page_queue = {layer: FreePageQueue(self.current_capacity) for layer in range(self.num_layers)}

    def allocate_page(self, layer: int, page_id: str) -> int:
        """
        分配指定层的页面
        """
        if page_id in self.page_map[layer]:
            self.page_usage[layer][self.page_map[layer][page_id]] += 1
            return self.page_map[layer][page_id]

        page_idx = self.free_page_queue[layer].dequeue_left()
        if page_idx == -1:
            if self.current_capacity >= self.max_pages:
                logging.error(f"Layer {layer}: Page pool has reached maximum capacity.")
                return -1
            self._expand_pool(layer)
            return self.allocate_page(layer, page_id)

        assert self.page_usage[layer][page_idx] == 0, "Page usage count should be 0 before allocation."
        self.page_usage[layer][page_idx] += 1
        self.page_map[layer][page_id] = page_idx
        return page_idx

    def _expand_pool(self, layer: int):
        """
        为指定层扩展页面池
        """
        if self.current_capacity >= self.max_pages:
            logging.info(f"Layer {layer}: Page pool has reached maximum capacity.")
            return

        # 混合策略扩展步长
        expansion_step = self.expansion_step if self.current_capacity <= 256 else min(16, self.max_pages - self.current_capacity)
        new_capacity = min(self.current_capacity + expansion_step, self.max_pages)
        additional_pages = new_capacity - self.current_capacity

        # 扩展 k_cache 和 v_cache 的页面池
        new_k_cache_pages = torch.zeros((additional_pages, self.num_head, self.headsize // self.x_factor, self.block_size, self.x_factor),
                                        device=self.device, dtype=self.dtype)
        new_v_cache_pages = torch.zeros((additional_pages, self.num_head, self.headsize, self.block_size),
                                        device=self.device, dtype=self.dtype)
        self.k_cache_pool[layer] = torch.cat((self.k_cache_pool[layer], new_k_cache_pages), dim=0)
        self.v_cache_pool[layer] = torch.cat((self.v_cache_pool[layer], new_v_cache_pages), dim=0)

        self.page_usage[layer].extend([0] * additional_pages)
        self.free_page_queue[layer].batch_enqueue_right(range(self.current_capacity, new_capacity))
        self.current_capacity = new_capacity

        if self.current_capacity <= 256:
            self.expansion_step *= 2

        logging.info(f"Layer {layer}: Expanded page pool to new capacity: {self.current_capacity}, expansion_step: {expansion_step}")

    def hash_block(self, block_token_idx: List[str], block_token_pos: List[str]) -> str:
        hashed_block_token_pos = '#'.join(block_token_pos)
        return '#'.join(block_token_idx)+"@"+hashed_block_token_pos

    def replace_first_masked_position(self, logical_id: str, token_id: int) -> Tuple[str, int]:
        # 将字符串分成两部分
        part1, part2 = logical_id.split('@')
        
        # 处理前一部分
        parts = part1.split('#')
        if 'M' in parts:
            first_masked_pos = parts.index('M')  # 找到 'M' 的位置
            parts[first_masked_pos] = str(token_id)  # 替换 'M' 为 token_id
            part1 = '#'.join(parts)  # 更新前一部分
        else:
            raise ValueError("No masked position ('M') found in the first part of the logical_id")
        
        # 拼接处理后的部分和未处理的后部分
        updated_logical_id = f"{part1}@{part2}"
        return updated_logical_id, first_masked_pos

    def get_cached_pages(self, layer:int):
        return self.k_cache_pool[layer], self.v_cache_pool[layer]
    
    def print_page_Tensor(self, layer):
        if len(self.sequence_page_table[layer]) == 0:
            print("sequence_page_table is empty")
            return
        print("print k_cache_pool")
        print(self.k_cache_pool[layer][self.sequence_page_table[layer][0]])
        print("print v_cache_pool")
        print(self.v_cache_pool[layer][self.sequence_page_table[layer][0]])

    def prefill(self, layer: int, token_idx_list: list, k_data: torch.Tensor, v_data: torch.Tensor):
        """
        预填充页面池，根据 token_idx_list 划分块并将数据填充到页面中。
        映射表 str(token_idx_list) -- > 物理块id
        Args:
            token_idx_list (list): 要处理的 token 索引列表。
            data (Tensor): 形状为 [b, seq_len, num_head, head_size] 的数据张量，用于填充页面池中的相应块。
        # q : 为什么不以块的为单位进行重用，因为有位置编码，如果把位置编码融合进attention kernel中呢
        """
        assert k_data is not None and v_data is not None, "k_data and v_data must be provided for prefill"
        b, seq_len, num_head, head_size = k_data.shape
        b_v, seq_len_v, num_head_v, head_size_v = v_data.shape
        assert b == 1, "Only support batch size of 1 for prefill"
        assert (b==b_v) & (seq_len==seq_len_v) & (num_head==num_head_v) & (head_size==head_size_v), "k, v shape mush same"
        assert head_size % self.x_factor == 0, "head_size must be divisible by self.x_factor"

        k_data = k_data.squeeze(0) # [seq_len, num_head, head_size]
        v_data = v_data.squeeze(0) # [seq_len, num_head, head_size]
        reshaped_k_tensor = k_data.contiguous().view(seq_len, num_head, head_size // self.x_factor, self.x_factor)
        k_data = reshaped_k_tensor.permute(1, 2, 0, 3) # [num_head, head_size // x_factor, seq_len, x_factor]
        v_data = v_data.permute(1, 2, 0) # [num_head, head_size, seq_len]
        
        # 划分 token_idx_list 为多个小块
        blocks = [list(map(str, token_idx_list[i:i + self.block_size])) for i in range(0, len(token_idx_list), self.block_size)]

        # 检查最后一个块的大小，如果小于 block_size，则进行填充
        if len(blocks[-1]) < self.block_size:
            blocks[-1].extend(['M'] * (self.block_size - len(blocks[-1])))

        # 为每个块分配页面并填充
        for idx, block in enumerate(blocks):
            block_token_pos = [str(i) for i in range(idx*self.block_size, idx*self.block_size+self.block_size)]
            logical_page_id = self.hash_block(block, block_token_pos)
            mask_count = logical_page_id.count('M')
            if 'M' in logical_page_id:
                self.remain_pages[layer].append(logical_page_id)
            physical_page_idx = self.allocate_page(layer, logical_page_id)
            if physical_page_idx == -1:
                raise ValueError("Allocate memory out of max page size, Failed to allocate page for block:", block)
            fill_token_size = self.block_size - mask_count
            self.sequence_page_table[layer].extend([physical_page_idx] * fill_token_size)
            # 对于每个块中的每个 token_idx，从 data 中提取对应的 token embedding
            start_pos = idx * self.block_size
            end_pos = min(start_pos + self.block_size, seq_len)
            self.k_cache_pool[layer][physical_page_idx, :, :, :end_pos-start_pos, :] = k_data[:, :, start_pos:end_pos, :]
            self.v_cache_pool[layer][physical_page_idx, :, :, :end_pos-start_pos] = v_data[:, :, start_pos:end_pos]
    
    def decode(self, layer: int, block_token_idx:List[int], k_data: torch.Tensor, v_data: torch.Tensor):
    
        assert k_data is not None and v_data is not None, "data must be provided for decode"
        b, seq_len, num_head, head_size = k_data.shape
        b_v, seq_len_v, num_head_v, head_size_v = v_data.shape
        assert b == 1 and seq_len==1, "Only support batch size of 1 for prefill"
        assert (b==b_v) & (seq_len==seq_len_v) & (num_head==num_head_v) & (head_size==head_size_v), "k, v shape mush same"
        assert head_size % self.x_factor == 0, "head_size must be divisible by self.x_factor"

        k_data = k_data.squeeze(0).squeeze(0) # [num_head, head_size]
        v_data = v_data.squeeze(0).squeeze(0) # [num_head, head_size]
        k_data = k_data.contiguous().view(num_head, head_size // self.x_factor, self.x_factor)


        current_token_id = block_token_idx[-1]
        # 如果有未填满的page
        if self.remain_pages[layer]:
            # 找到未填满的page, 现在只考虑单个sequence，所以remain_pages只会有一个元素
            # 并且该元素一定指向先前生成token的尾端
            logical_remain_page_id = self.remain_pages[layer][0]
            # 替换其中首个'M'字段，组成新的逻辑页id, 并查找第一个 'M' 在列表中的位置
            new_page_id, first_masked_pos = self.replace_first_masked_position(logical_remain_page_id, current_token_id)
            # 找到当前物理页id
            physical_current_page_id = self.page_map[layer][logical_remain_page_id]
            # 是否已经有对应的page -- 如果有, 直接复用, 在共享prompt和parallel sampling中，可能会出现该情况 
            if new_page_id in self.page_map[layer]:
                # 找到更新的物理页
                physical_update_page_id = self.page_map[layer][new_page_id]
                # 当前引用更新，不再引用旧的物理页
                self.page_usage[layer][physical_current_page_id] -= 1
                self.page_usage[layer][physical_update_page_id] += 1

                # 找到先前生成token有多少个mask，只考虑单个sequence，走入该分支，一定是1
                mask_count = logical_remain_page_id.count('M')
                # 对应可以找到该page填充的token数目
                fill_token_size = self.block_size - mask_count
                # 改写先前对应token的映射
                self.sequence_page_table[layer][-fill_token_size:] = [physical_update_page_id] * fill_token_size
                # 将当前生成token对应的物理页添加到page table中
                self.sequence_page_table[layer].append(physical_update_page_id)

                # 检查先前的物理页是否有别的引用,如果没有
                if self.page_usage[layer][physical_current_page_id] == 0:
                   # 将先前的物理页放入空闲物理页队列中
                   self.free_page_queue[layer].enqueue_right(physical_current_page_id)
                   # 删除掉对应的逻辑页--物理页映射
                   del self.page_map[layer][logical_remain_page_id]
                   del self.remain_pages[layer][0]
            else: # 如果没有, 往未填满的page中插入
                # 将数据插入page pool对应的位置当中
                self.k_cache_pool[layer][self.page_map[layer][logical_remain_page_id], :, :, first_masked_pos, :] = k_data
                self.v_cache_pool[layer][self.page_map[layer][logical_remain_page_id], :, :, first_masked_pos] = v_data
                # 删除旧的逻辑页面id -- 物理页id映射，因为逻辑页面id要更换
                del self.page_map[layer][logical_remain_page_id]
                # 更新逻辑页id的映射
                self.page_map[layer][new_page_id] = physical_current_page_id
                self.sequence_page_table[layer].append(physical_current_page_id)
                # 说明当前逻辑页已被填满
                if new_page_id.count('M') == 0:
                    del self.remain_pages[layer][0]
        else: # 如果没有，表示所有的物理页均已填满，要重新申请一个物理页
            # 设置新申请物理页的逻辑页id
            current_block_id = [str(current_token_id)] + ['M'] * (self.block_size - 1)
            # 当前的token所处位置减一，一定可以整除block_size
            block_token_pos = [str(i) for i in range(len(block_token_idx)-1, len(block_token_idx)-1+self.block_size)]
            logical_page_id = self.hash_block(current_block_id, block_token_pos)

            # 将对应的逻辑页追加到remain_pages中去
            if 'M' in logical_page_id:
                self.remain_pages[layer].append(logical_page_id)
            # 申请物理页
            physical_page_idx = self.allocate_page(layer, logical_page_id)
            if physical_page_idx == -1:
                raise ValueError("Allocate memory out of max page size, Failed to allocate page for block:", physical_page_idx)
            # 将数据插入page pool对应的位置当中
            self.k_cache_pool[layer][physical_page_idx, :, :, 0, :] = k_data
            self.v_cache_pool[layer][physical_page_idx, :, :, 0] = v_data
            self.sequence_page_table[layer].append(physical_page_idx)
            

# Helper function to initialize manager
def init_manager(block_size=2, num_head=2, headsize=4, initial_pages=2, max_pages=16, dtype=torch.float32, x_factor=2, num_layers=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return PageTableManager(block_size=block_size, num_head=num_head, headsize=headsize,
                            initial_pages=initial_pages, max_pages=max_pages, dtype=dtype,
                            x_factor=x_factor, num_layers=num_layers), device

def init_tensor(token_idx_list, init_one=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size, embedding_dim = 10, 8
    if init_one:
        embeddings_k = torch.ones(vocab_size, embedding_dim, device=device, dtype=torch.float32)
        embeddings_v = torch.ones(vocab_size, embedding_dim, device=device, dtype=torch.float32)
    else:
        embeddings_k = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.float32)
        embeddings_v = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.float32)
    data_k = embeddings_k[token_idx_list].view(1, len(token_idx_list), 2, 4)
    data_v = embeddings_v[token_idx_list].view(1, len(token_idx_list), 2, 4)
    return data_k, data_v

def test_cpoy():
    manager, device = init_manager()
    layer = 1
    token_idx_list = [1, 2, 3, 4, 5]
    k_data, v_data = init_tensor(token_idx_list)
    manager.prefill(layer, token_idx_list, k_data, v_data)
    
    print(f"Original data before update: {manager.k_cache_pool[layer][0,0,0,0,0]}")
    k_cache, _ = manager.get_cached_pages(layer)
    k_cache[0,0,0,0,0] = 999
    print(f"Modified data: {k_cache[0,0,0,0,0]}")
    print(f"Original data after update: {manager.k_cache_pool[layer][0,0,0,0,0]}")


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
    manager, _ = init_manager(initial_pages=5, max_pages=10, num_layers=3)
    layer = 0
    page_id = "page1"
    allocated_page = manager.allocate_page(layer, page_id)
    assert allocated_page != -1, "Page allocation failed: No valid page allocated"
    assert manager.page_map[layer][page_id] == allocated_page, "Page allocation failed: Incorrect page mapping"
    print("allocate_page test passed for layer.")

def test_prefill():
    manager, device = init_manager()
    token_idx_list = [1, 2, 3, 4, 5, 6]
    k_data, v_data = init_tensor(token_idx_list, init_one=True)
    layer = 1
    manager.prefill(layer, token_idx_list, k_data, v_data)
    
    for page_id, idx in manager.page_map[layer].items():
        assert (manager.k_cache_pool[layer][idx] == 1).all(), f"Prefill failed: k_cache_pool Data mismatch on page {page_id}"
        assert (manager.v_cache_pool[layer][idx] == 1).all(), f"Prefill failed: v_cache_pool Data mismatch on page {page_id}"
    print("Prefill test passed for layer.")

def test_decode_fill_page():
    torch.manual_seed(1999)
    manager, device = init_manager()
    layer = 1
    token_idx_list = [1, 2, 3, 4, 5]
    k_data, v_data = init_tensor(token_idx_list)
    manager.prefill(layer, token_idx_list, k_data, v_data)
    
    print(manager.sequence_page_table[layer])
    
    new_token_idx = [1, 2, 3, 4, 5, 6]
    token_data_k = torch.randn((2, 4), device=device, dtype=torch.float32)
    token_data_v = torch.randn((2, 4), device=device, dtype=torch.float32)
    token_data_k = token_data_k.view(1, 1, 2, 4)
    token_data_v = token_data_v.view(1, 1, 2, 4)
    manager.decode(layer, new_token_idx, token_data_k, token_data_v)

    print(manager.sequence_page_table[layer])
    
    logical_page_id = "5#6@4#5"
    physical_page_idx = manager.page_map[layer].get(logical_page_id, -1)
    
    token_data_k = token_data_k.view(2, 2, 2)
    token_data_v = token_data_v.view(2, 4)
    
    assert physical_page_idx != -1, "Decode failed: No allocated page for new token."
    assert (manager.k_cache_pool[layer][physical_page_idx, :, :, 1, :] == token_data_k).all(), "Decode failed: Data mismatch on decoded page"
    assert (manager.v_cache_pool[layer][physical_page_idx, :, :, 1] == token_data_v).all(), "Decode failed: Data mismatch on decoded page"
    print("Decode fill remain page test passed for layer.")

def test_decode_reuse():
    torch.manual_seed(1999)
    manager, device = init_manager()
    layer = 1
    vocab_size, embedding_dim = 10, 4
    embeddings = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.float32)
    token_idx_list = [1, 2, 3, 4, 3]
    k_data, v_data = init_tensor(token_idx_list)
    manager.prefill(layer, token_idx_list, k_data, v_data)
    print(manager.sequence_page_table[layer])
    
    new_token_id = 4
    token_idx_list.append(new_token_id)
    token_data_k = torch.randn((2, 4), device=device, dtype=torch.float32)
    token_data_v = torch.randn((2, 4), device=device, dtype=torch.float32)
    token_data_k = token_data_k.view(1, 1, 2, 4)
    token_data_v = token_data_v.view(1, 1, 2, 4)
    manager.decode(layer, token_idx_list, token_data_k, token_data_v)
    print(manager.sequence_page_table[layer])
    
    logical_page_id = "3#4@4#5"
    physical_page_idx = manager.page_map[layer].get(logical_page_id, -1)
    
    token_data_k = token_data_k.view(2, 2, 2)
    token_data_v = token_data_v.view(2, 4)
    
    assert physical_page_idx != -1, "Decode reuse failed: No valid page allocated."
    assert 2 not in manager.free_page_queue[layer].free_page_queue, "Decode reuse failed: Expected 2 not in manager.free_page_queue[layer].free_page_queue."
    assert (manager.k_cache_pool[layer][physical_page_idx, :, :, 1, :] == token_data_k).all(), "Decode failed: Data mismatch on decoded page"
    assert (manager.v_cache_pool[layer][physical_page_idx, :, :, 1] == token_data_v).all(), "Decode failed: Data mismatch on decoded page"
    print("Decode reuse test passed for layer.")


def test_decode_new_page():
    torch.manual_seed(1999)
    manager, device = init_manager()
    layer = 1
    vocab_size, embedding_dim = 10, 4
    embeddings = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.float32)
    token_idx_list = [1, 2, 3, 4]
    k_data, v_data = init_tensor(token_idx_list)
    manager.prefill(layer, token_idx_list, k_data, v_data)
    print(manager.sequence_page_table[layer])
    
    new_token_id = 5
    token_idx_list.append(new_token_id)
    token_data_k = torch.randn((2, 4), device=device, dtype=torch.float32)
    token_data_v = torch.randn((2, 4), device=device, dtype=torch.float32)
    token_data_k = token_data_k.view(1, 1, 2, 4)
    token_data_v = token_data_v.view(1, 1, 2, 4)
    manager.decode(layer, token_idx_list, token_data_k, token_data_v)
    print(manager.sequence_page_table[layer])

    logical_page_id = "5#M@4#5"
    physical_page_idx = manager.page_map[layer].get(logical_page_id, -1)
    
    token_data_k = token_data_k.view(2, 2, 2)
    token_data_v = token_data_v.view(2, 4)
    
    assert physical_page_idx != -1, "Decode failed: No valid page allocated."
    assert (manager.k_cache_pool[layer][physical_page_idx, :, :, 0, :] == token_data_k).all(), "Decode failed: Data mismatch on decoded page"
    assert (manager.v_cache_pool[layer][physical_page_idx, :, :, 0] == token_data_v).all(), "Decode failed: Data mismatch on decoded page"
    print("Decode new page test passed for layer.")


if __name__ == "__main__":
    """
    Testing the functionality of the page manager.
    """
    test_cpoy()
    test_free_page_queue()
    test_allocate_page()
    test_prefill()
    test_decode_fill_page()
    test_decode_reuse()
    test_decode_new_page()