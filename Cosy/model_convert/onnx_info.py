import os
import onnx


# 定义统计单个 ONNX 文件中的算子
def count_onnx_operators(file_path):
    # 加载 ONNX 模型
    onnx_model = onnx.load(file_path)

    # 初始化算子计数器
    operator_counts = {}

    # 遍历所有节点
    for node in onnx_model.graph.node:
        op_type = node.op_type
        if op_type in operator_counts:
            operator_counts[op_type] += 1
        else:
            operator_counts[op_type] = 1

    # 统计总算子数量
    total_operators = sum(operator_counts.values())

    return operator_counts, total_operators



def counter_dir():
    # 遍历当前目录下的所有 .onnx 文件
    onnx_files = [f for f in os.listdir(".") if f.endswith(".onnx")]

    # 逐个统计每个 ONNX 文件中的算子数量
    for onnx_file in onnx_files:
        operator_counts, total_operators = count_onnx_operators(onnx_file)

        # 输出结果
        print(f"\nFile: {onnx_file}")
        for op_type, count in operator_counts.items():
            print(f"  Operator: {op_type}, Count: {count}")
        print(f"  Total operators: {total_operators}")
        

def counter_file(file_path):
    
    operator_counts, total_operators = count_onnx_operators(file_path)

    # 输出结果
    print(f"\nFile: {file_path}")
    for op_type, count in operator_counts.items():
        print(f"  Operator: {op_type}, Count: {count}")
    print(f"  Total operators: {total_operators}")
    

if __name__ == "__main__":
    file_path = r"E:\Codes\CosyVoice\model_convert\onnx\llm_model_stage1.onnx"
    fil2 = r"E:\Codes\CosyVoice\model_convert\onnx\llm_model_stage1_optimized.onnx"
    counter_file(file_path)
    
    print("nvidia...............")
    counter_file(fil2)