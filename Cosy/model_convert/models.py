import json
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
import onnx_graphsurgeon as gs
import os
from polygraphy.backend.onnx.loader import fold_constants
import re
import tempfile
import torch
import torch.nn.functional as F
from cosyvoice.cli.cosyvoice import CosyVoice
from hyperpyyaml import load_hyperpyyaml

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        return gs.export_onnx(self.graph) if return_onnx else self.graph

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, 'model.onnx')
            onnx_inferred_path = os.path.join(temp_dir, 'inferred.onnx')
            onnx.save_model(onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def fuse_mha_qkv_int8_sq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()

        # mha  : fuse QKV QDQ nodes
        # mhca : fuse KV QDQ nodes
        q_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_q/input_quantizer/DequantizeLinear_output_0"
        )
        k_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_k/input_quantizer/DequantizeLinear_output_0"
        )
        v_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_v/input_quantizer/DequantizeLinear_output_0"
        )

        qs = list(sorted(map(
            lambda x: x.group(0),  # type: ignore
            filter(lambda x: x is not None, [re.match(q_pat, key) for key in keys]),
            )))
        ks = list(sorted(map(
            lambda x: x.group(0),  # type: ignore
            filter(lambda x: x is not None, [re.match(k_pat, key) for key in keys]),
            )))
        vs = list(sorted(map(
            lambda x: x.group(0),  # type: ignore
            filter(lambda x: x is not None, [re.match(v_pat, key) for key in keys]),
            )))

        removed = 0
        assert len(qs) == len(ks) == len(vs), "Failed to collect tensors"
        for q, k, v in zip(qs, ks, vs):
            is_mha = all(["attn1" in tensor for tensor in [q, k, v]])
            is_mhca = all(["attn2" in tensor for tensor in [q, k, v]])
            assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

            if is_mha:
                tensors[k].outputs[0].inputs[0] = tensors[q]
                tensors[v].outputs[0].inputs[0] = tensors[q]
                del tensors[k]
                del tensors[v]
                removed += 2
            else:  # is_mhca
                tensors[k].outputs[0].inputs[0] = tensors[v]
                del tensors[k]
                removed += 1
        print(f"Removed {removed} QDQ nodes")
        return removed # expected 72 for L2.5
        

class BaseModel():
    def __init__(self,
        model_dir,
        device='cuda',
        verbose=True,
        fp16=False,
        int8=False,
        max_batch_size=16,
    ):

        self.name = self.__class__.__name__
        self.device = device
        self.verbose = verbose

        self.fp16 = fp16
        self.int8 = int8

        self.model_dir = model_dir
        with open('{}/cosyvoice_onnx.yaml'.format(model_dir), 'r') as f:
            self.models = load_hyperpyyaml(f)

        self.min_batch = 1
        self.max_batch = max_batch_size


    def get_model(self, torch_inference=''):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, *args, **kwargs):
        pass


    # Helper utility for ONNX export
    def export_onnx(
        self,
        onnx_path,
        onnx_opt_path,
        onnx_opset,
        custom_model=None,
        custom_opsets=None,
        verbose=False,
        *args, 
        **kwargs
    ):
        onnx_opt_graph = None
        # Export optimized ONNX model (if missing)
        if not os.path.exists(onnx_opt_path):
            if not os.path.exists(onnx_path):
                print(f"[I] Exporting ONNX model: {onnx_path}")
                def export_onnx(model):
                    inputs = self.get_sample_input(*args, **kwargs)
                    torch.onnx.export(model,
                        inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=onnx_opset,
                        do_constant_folding=True,
                        input_names=self.get_input_names(),
                        output_names=self.get_output_names(),
                        dynamic_axes=self.get_dynamic_axes(),
                        custom_opsets=custom_opsets,  # 使用自定义的 opset 域和版本
                        verbose=verbose
                    )
                if custom_model:
                    with torch.inference_mode():
                        export_onnx(custom_model)
                else:
                    with torch.inference_mode(), torch.autocast("cuda"):
                        export_onnx(self.get_model())
            else:
                print(f"[I] Found cached ONNX model: {onnx_path}")

            print(f"[I] Optimizing ONNX model: {onnx_opt_path}")
            onnx_opt_graph = self.optimize(onnx.load(onnx_path))
            if onnx_opt_graph.ByteSize() > 2147483648:
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_opt_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False)
            else:
                onnx.save(onnx_opt_graph, onnx_opt_path)
        else:
            print(f"[I] Found cached optimized ONNX model: {onnx_opt_path} ")
            
    
    def get_model_object(self, models, obj_name):
        return models.get(obj_name)
            
    
    def load_partial_model(self, model_name, obj_name):
        # Get the LLM object
        model = self.get_model_object(self.models, obj_name)

        # Load the full state dict from the checkpoint
        checkpoint_state_dict = torch.load(f'{self.model_dir}/{model_name}.pt', map_location=self.device)

        # Get the current model's state dict
        model_state_dict = model.state_dict()

        # Create a new state dict that will contain only the matching parameters
        filtered_state_dict = {}

        # Iterate over the model's state dict keys
        for key in model_state_dict.keys():
            if key in checkpoint_state_dict and model_state_dict[key].shape == checkpoint_state_dict[key].shape:
                # Add matching parameter to the filtered state dict
                filtered_state_dict[key] = checkpoint_state_dict[key]
            else:
                print(f"Skipping {key}: shape mismatch or not found in checkpoint")

        # Load the filtered state dict into the model
        model.load_state_dict(filtered_state_dict, strict=False)

        return model


    def optimize(self, onnx_graph, return_onnx=True, **kwargs):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        if kwargs.get('fuse_mha_qkv_int8', False):
            opt.fuse_mha_qkv_int8_sq()
            opt.info(self.name + ': fuse QKV nodes')
        onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
        opt.info(self.name + ': finished')
        return onnx_opt_graph


class LLMModelStage1(BaseModel):
    def __init__(self,
        model_dir,
        device,
        verbose,
        max_batch_size,
        fp16=False
    ):
        super(LLMModelStage1, self).__init__(model_dir=model_dir, device=device, verbose=verbose, fp16=fp16, max_batch_size=max_batch_size)

    def get_model(self):
        model = self.load_partial_model("llm", 'llm_stage1')
        model.to(self.device)
        if self.fp16:
            model.half()
        return model
    

    def get_input_names(self):
        return [
            'text',              # Text input
            'text_len',                # Text length input
            'prompt_text',             # Prompt text input
            'prompt_text_len',         # Prompt text length input
            'prompt_speech_token',     # Prompt speech tokens input
            'prompt_speech_token_len'  # Prompt speech tokens length input
            'embedding',               # Embedding input 
        ]


    # speech_token.shape: torch.Size([1, 227])  -- speech_token.shape: torch.Size([1, 559])
    def get_output_names(self):
        return ['lm_input', "text_len"]

    def get_dynamic_axes(self):
        return {
            # Inputs
            'text': {1: 'L_text'},  # Batch and sequence length (text)  
            'prompt_text': {1: 'L_prompt_text'},  # Batch and sequence length (prompt text)
            'prompt_speech_token': {1: 'L_prompt_speech'},  # Batch and sequence length (prompt speech tokens)
            
            # Outputs
            'lm_input': {1: 'L_text'},  # Batch and sequence length (text input)
        }


    def get_sample_input(self, batch_size=1):
        """
        生成用于模型推理的样本输入数据，包括文本、提示文本、语音 token 和嵌入向量等。

        Args:
            batch_size (int): 批次大小。
            static_shape (bool): 是否使用静态形状，默认为True。如果为False，将对形状进行调整。
            sampling (int): 采样次数，默认为25。
            max_token_text_ratio (float): 最大token文本比率，默认为20。
            min_token_text_ratio (float): 最小token文本比率，默认为2。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            返回一组样本输入数据。

        # ref shape
        text: torch.Tensor,   --  torch.Size([1, 65])  -- torch.Size([1, 22])
        text_len: torch.Tensor,  -- torch.Size([1])  -- torch.Size([1])
        prompt_text: torch.Tensor, -- torch.Size([1, 16])  -- torch.Size([1, 36])
        prompt_text_len: torch.Tensor, -- torch.Size([1])  -- torch.Size([1])
        prompt_speech_token: torch.Tensor, -- torch.Size([1, 174]) -- torch.Size([1, 302])
        prompt_speech_token_len: torch.Tensor, -- torch.Size([1]) -- torch.Size([1])
        embedding: torch.Tensor, -- torch.Size([1, 192]) -- torch.Size([1, 192])
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        """

        # 确定数据类型
        dtype = torch.float16 if self.fp16 else torch.float32

        text = torch.randint(0, 100, (1, 65), dtype=torch.int32, device=self.device)
        text_len = torch.randint(1, 66, (1,), dtype=torch.int32, device=self.device)
        prompt_text = torch.randint(0, 100, (1, 16), dtype=torch.int32, device=self.device)
        prompt_text_len = torch.randint(1, 17, (1,), dtype=torch.int32, device=self.device)
        prompt_speech_token = torch.randint(0, 100, (1, 174), dtype=torch.int32, device=self.device)
        prompt_speech_token_len = torch.randint(1, 175, (1,), dtype=torch.int32, device=self.device)
        embedding = torch.randn(1, 192, dtype=dtype, device=self.device)

        return (
            text,
            text_len,
            prompt_text,
            prompt_text_len,
            prompt_speech_token,
            prompt_speech_token_len,
            embedding,
        )



class LLMModelStage2(BaseModel):
    def __init__(self,
        model_dir,
        device,
        verbose,
        max_batch_size,
        fp16=False
    ):
        super(LLMModelStage2, self).__init__(model_dir=model_dir, device=device, verbose=verbose, fp16=fp16, max_batch_size=max_batch_size)

    def get_model(self):
        model = self.load_partial_model("llm", 'llm_stage2')
        model.to(self.device)
        if self.fp16:
            model.half()
        return model
        

    def get_input_names(self):
        return [
            'lm_input',             
            'att_cache',           
            'cnn_cache',            
        ]


    # speech_token.shape: torch.Size([1, 227])  -- speech_token.shape: torch.Size([1, 559])
    def get_output_names(self):
        return ['logp', 'att_cache_out', 'cnn_cache_out']

    '''
    logp shape torch.Size([1, 4097]) -- 4097 fixed
    logp dtype torch.float16
    '''
    def get_dynamic_axes(self):
        return {
            # Inputs
            'lm_input': {1: 'L_text'},  # Batch and sequence length (text input)
            'att_cache': {0: 'S', 1: 'H', 2: 'L_cache', 3: 'D_cache'},  # Batch, heads, sequence length, cache dimension
            'cnn_cache': {0: 'S', 1: 'H', 2: 'L_cache', 3: 'D_cache'},  # Batch, heads, sequence length, cache dimension

            # Outputs
            'att_cache_out': {0: 'S', 1: 'H', 2: 'L_cache', 3: 'D_cache'},  # Batch, heads, sequence length, cache dimension
            'cnn_cache_out': {0: 'S', 1: 'H', 2: 'L_cache', 3: 'D_cache'},
        }


    def get_sample_input(self, batch_size=1):
        """
        Generate sample input data for model inference, including text input, prompt text, 
        speech tokens, and embedding vectors.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            Returns a set of sample input data.
        """

        # Determine data type
        
        dtype = torch.float16 if self.fp16 else torch.float32

        # Define static shapes based on the provided reference shapes
        time = 363
        mel_dim = 1024
        hidden_dim = 1024
        num_layers = 14
        num_heads = 16
        cache_t1 = 0
        cache_t2 = 0
        d_k = hidden_dim // num_heads

        # 示例输入张量

        # 1. lm_input: chunk input (b=1, time, mel-dim)
        lm_input = torch.randn(1, time, mel_dim, dtype=dtype, device=self.device)

        # 2. att_cache: cache tensor for KEY & VALUE
        att_cache = torch.randn(num_layers, num_heads, cache_t1, d_k * 2, dtype=dtype, device=self.device)

        # 3. cnn_cache: cache tensor for cnn_module
        cnn_cache = torch.randn(num_layers, 1, hidden_dim, cache_t2, dtype=dtype, device=self.device)

        # Return the generated sample input data
        return lm_input, att_cache, cnn_cache



class FlowModelStage1(BaseModel):
    def __init__(self,
        model_dir,
        device,
        verbose,
        max_batch_size,
        fp16=False
    ):
        super(FlowModelStage1, self).__init__(model_dir=model_dir, device=device, verbose=verbose, fp16=fp16, max_batch_size=max_batch_size)

    def get_model(self):
        model = self.load_partial_model("flow", 'flow_stage1')
        model.to(self.device)
        if self.fp16:
            model.half()
        return model
    

    def get_input_names(self):
        return [
            'token',             
            'token_len',           
            'token_len1', 
            'token_len2',
            'prompt_feat', 
            'mask'          
        ]


    # speech_token.shape: torch.Size([1, 227])  -- speech_token.shape: torch.Size([1, 559])
    def get_output_names(self):
        return ['h', 'conds']


    def get_dynamic_axes(self):
        return {
            # Inputs
            'token': {0: 'L'}, 
            'token_len': {0: 'L'}, 
            'prompt_feat': {1: 'L'},  
            'mask': {1: 'L'},

            # Outputs
            'h': {1: 'L'},  
            'conds': {2: 'L'}  
        }


    def get_sample_input(self, batch_size=1):

        # Determine data type
        '''
        token,
        token_len,
        token_len1,
        token_len2,
        prompt_feat,
        mask
        h: torch.Size([1, 1230, 80])  fp32
        conds: torch.Size([1, 80, 1230]) fp32
        
        h: torch.Size([1, 883, 80])
        conds: torch.Size([1, 80, 883])
        '''

        token_dtype = torch.int64
        prompt_feat_dtype = torch.float32
        mask_dtype = torch.float32

        # 设置张量的形状
        token_shape = (1, 513)  # 或 (541,) 根据需要调整
        prompt_feat_shape = (1, 520, 80)  # 或 (1, 299, 80)
        mask_shape = (1, 513, 1)  # (1, 715, 1)

        # 生成张量
        token = torch.randint(low=0, high=1000, size=token_shape, dtype=token_dtype, device=self.device)
        token_len = torch.tensor([token_shape[1]], dtype=torch.int32)
        
        token_len1 = 302
        token_len2 = 211
        
        prompt_feat = torch.randn(prompt_feat_shape, dtype=prompt_feat_dtype, device=self.device)
        mask = torch.randn(mask_shape, dtype=mask_dtype, device=self.device)

        return token, token_len, token_len1, token_len2, prompt_feat, mask



class FlowModelStage2(BaseModel):
    def __init__(self,
        model_dir,
        device,
        verbose,
        max_batch_size,
        fp16=False
    ):
        super(FlowModelStage2, self).__init__(model_dir=model_dir, device=device, verbose=verbose, fp16=fp16, max_batch_size=max_batch_size)

    def get_model(self):
        model = self.load_partial_model("flow", 'flow_stage2')
        model.to(self.device)
        if self.fp16:
            model.half()

        return model
    

    def get_input_names(self):
        return [
            'h',             
            'mask',           
            'conds',
            'mel_len1', 
            'embedding'       
        ]


    # speech_token.shape: torch.Size([1, 227])  -- speech_token.shape: torch.Size([1, 559])
    def get_output_names(self):
        return ['feat']

    '''
    feat.shape: torch.Size([1, 80, 935])
    feat.dtype: torch.float32
    feat.shape: torch.Size([1, 80, 377])
    feat.dtype: torch.float32
    '''
    def get_dynamic_axes(self):
        return {
            # Inputs
            'h': {1: 'L'},  
            'mask': {1: 'L'},  
            'conds': {1: 'L'},  

            # Outputs
            # Assuming 'logp' is an output with variable sequence length
            'feat': {2: 'L'}  # Batch size & feat length
        }


    def get_sample_input(self, batch_size=1):
        
        
        '''
        h, mask, conds, mel_len1, embedding
        '''

        # Determine data type

        h = torch.zeros(1, 883, 80, dtype=torch.float32)  # 形状为[1, 883, 80]的全0张量
        mask = torch.zeros(1, 883, dtype=torch.float32)  # 形状为[1, 883]的全0张量
        conds = torch.zeros(1, 80, 883, dtype=torch.float32)  # 形状为[1, 80, 883]的全0张量
        embedding = torch.zeros(1, 80, dtype=torch.float32)  # 形状为[1, 80]的全0张量

        mel_len1 = 42

        return h, mask, conds, mel_len1, embedding




class HifiModel(BaseModel):
    def __init__(self,
        model_dir,
        device,
        verbose,
        max_batch_size,
        fp16=False
    ):
        super(HifiModel, self).__init__(model_dir=model_dir, device=device, verbose=verbose, fp16=fp16, max_batch_size=max_batch_size)

    def get_model(self):
        model = self.load_partial_model("hift", 'hift')
        model.to(self.device)
        if self.fp16:
            model.half()
        #traced_model = torch.jit.trace(model, self.get_sample_input())
        #print(traced_model.graph)
        return model
    

    def get_input_names(self):
        return [
            'mel',             
            's_stft'     
        ]


    # speech_token.shape: torch.Size([1, 227])  -- speech_token.shape: torch.Size([1, 559])
    def get_output_names(self):
        return ['magnitude', 'phase']


    def get_dynamic_axes(self):
        return {
            # Inputs
            'mel': {2: 'w_size'},  
            's_stft': {2: 'stft_sz'},  

            # Outputs
            # Assuming 'logp' is an output with variable sequence length
            'magnitude': {2: 's2'} ,
            'phase': {2: 's2'} 
        }


    def get_sample_input(self, batch_size=1):
        
        '''
        mel_shape: torch.Size([1, 80, 377])
        s_stft_shape: torch.Size([1, 18, 24129])
        
        magnitude_shape: torch.Size([1, 9, 24129])
        phase_shape: torch.Size([1, 9, 24129])
        
        mel_shape: torch.Size([1, 80, 935])
        s_stft_shape: torch.Size([1, 18, 59841])
        
        magnitude_shape: torch.Size([1, 9, 59841])
        phase_shape: torch.Size([1, 9, 59841])
        
        fp32
        '''

        # Determine data type

        mel_dtype = torch.float32
        s_stft_dtype = torch.float32

        # 设置张量的形状
        mel_shape = (1, 80, 377)  
        s_stft_shape = (1, 18, 24129)  

        # 生成张量
        mel = torch.randn(mel_shape, dtype=mel_dtype, device=self.device)
        s_stft = torch.randn(s_stft_shape, dtype=s_stft_dtype, device=self.device)
        
        return mel, s_stft


    