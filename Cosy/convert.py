from model_convert.models import LLMModelStage1, LLMModelStage2, FlowModelStage1, FlowModelStage2, HifiModel
import os


current_file_directory = os.path.dirname(os.path.abspath(__file__))
onnx_directory = os.path.join(current_file_directory, "model_convert", "onnx")


os.makedirs(onnx_directory, exist_ok=True)
print(f"Using ONNX directory: {onnx_directory}")

# 通用的模型导出函数
def export_model(model_instance, model_name, model_path, onnx_directory, onnx_opset=20):
    try:
        print(f"Exporting {model_name} for {model_path}...")
        onnx_path = os.path.join(onnx_directory, f"{model_name}.onnx")
        onnx_opt_path = os.path.join(onnx_directory, f"{model_name}_optimized.onnx")
        
        model_instance.export_onnx(
            onnx_path=onnx_path,
            onnx_opt_path=onnx_opt_path,
            onnx_opset=onnx_opset,
            custom_model=None,
            verbose=False
        )
        print(f"Exported {model_name} successfully to {onnx_path}")
    except Exception as e:
        print(f"Failed to export {model_name}: {e}")

# LLM 模型导出
def llm_model_convert1(model_path):
    model_dir = os.path.join(current_file_directory, model_path)
    
    llm_model_1 = LLMModelStage1(model_dir=model_dir, device='cpu', verbose=True, max_batch_size=16, fp16=False)
    export_model(llm_model_1, "llm_model_stage1", model_path, onnx_directory)


def llm_model_convert2(model_path):
    model_dir = os.path.join(current_file_directory, model_path)

    llm_model_2 = LLMModelStage2(model_dir=model_dir, device='cpu', verbose=True, max_batch_size=16, fp16=False)
    export_model(llm_model_2, "llm_model_stage2", model_path, onnx_directory)

# Flow 模型导出
def flow_model_stage1_convert(model_path):
    model_dir = os.path.join(current_file_directory, model_path)
    
    flow_model_stage1 = FlowModelStage1(model_dir=model_dir, device='cpu', verbose=True, max_batch_size=16, fp16=False)
    export_model(flow_model_stage1, "flow_model_stage1", model_path, onnx_directory)
    

def flow_model_stage2_convert(model_path):
    model_dir = os.path.join(current_file_directory, model_path)
    
    flow_model_stage2 = FlowModelStage2(model_dir=model_dir, device='cpu', verbose=True, max_batch_size=16, fp16=False)
    export_model(flow_model_stage2, "flow_model_stage2", model_path, onnx_directory)

# Hifi 模型导出
def hift_model_convert(model_path):
    model_dir = os.path.join(current_file_directory, model_path)
    
    hift_model = HifiModel(model_dir=model_dir, device='cpu', verbose=True, max_batch_size=16, fp16=False)
    export_model(hift_model, "hift_model", model_path, onnx_directory)

if __name__ == '__main__':
    model_path = "pretrained_models/CosyVoice-300M"
    #llm_model_convert1(model_path)
    #llm_model_convert2(model_path)
    flow_model_stage1_convert(model_path)
    #flow_model_stage2_convert(model_path)
    #hift_model_convert(model_path)
