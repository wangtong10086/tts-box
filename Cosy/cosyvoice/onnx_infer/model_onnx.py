import torch
import numpy as np
import uuid
from cosyvoice.utils.common import fade_in_out
import logging
import onnxruntime as ort
from typing import List
import typing as tp
from torch.distributions.uniform import Uniform

import torch.nn as nn
from torch.nn.utils import weight_norm
from cosyvoice.utils.common import ras_sampling_onnx
from scipy.signal import get_window
from cosyvoice.utils.mask import make_pad_mask
from hyperpyyaml import load_hyperpyyaml
from model_convert.models import LLMModelStage1, LLMModelStage2, FlowModelStage1, FlowModelStage2, HifiModel
import random


class CosyVoiceModel:

    def __init__(self,
                 llm_model_stage1_path: str,
                 llm_model_stage2_path: str,
                 flow_model_stage1_path: str,
                 flow_model_stage2_path: str,
                 hift_model_diir: str
                ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'] 
        print("init llm_stage1_session")
        self.llm_stage1_session = ort.InferenceSession(llm_model_stage1_path, providers=self.providers)
        print("init llm_stage2_session")
        self.llm_stage2_session = ort.InferenceSession(llm_model_stage2_path, providers=self.providers)
        print("init flow_stage1_session")
        self.flow_stage1_session = ort.InferenceSession(flow_model_stage1_path, providers=self.providers)
        print("init flow_stage2_session")
        self.flow_stage2_session = ort.InferenceSession(flow_model_stage2_path, providers=self.providers)
        print("init hift")
        with open('{}/hift.yaml'.format(hift_model_diir), 'r') as f:
           self.hift = load_hyperpyyaml(f)['hift']
        self.hift.load_state_dict(torch.load('{}/hift.pt'.format(hift_model_diir), map_location=self.device))
        self.hift.to(self.device).eval()
        
        self.token_min_hop_len = 100
        self.token_max_hop_len = 200
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = 34
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.hift_cache_dict = {}
        
        self.flow_output_size = 80
        self.speech_token_size = 4096
        self.llm_input_size = 1024
        self.sampling = ras_sampling_onnx
        
        
        
    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids


    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        sampling = 25
        max_token_text_ratio = 20
        min_token_text_ratio = 2

        # 定义输入的长度
        prompt_text_len = np.array([prompt_text.shape[1]], dtype=np.int32)
        text_len = np.array([text.shape[1]], dtype=np.int32)
        
        #print_variable_info(text=text, prompt_text=prompt_text, llm_prompt_speech_token=llm_prompt_speech_token, llm_embedding=llm_embedding)

        # 准备 Stage 1 输入
        inputs_stage1 = {
            "text": text.cpu().numpy(),  # Moving to CPU and converting to NumPy
            "text_len_input": text_len,  # Assuming text_len is already on the CPU
            "prompt_text": prompt_text.cpu().numpy(),  # Moving to CPU and converting to NumPy
            "prompt_text_len": prompt_text_len,  # Assuming this is already on the CPU
            "prompt_speech_token": llm_prompt_speech_token.cpu().numpy(),  # Moving to CPU and converting to NumPy
            "embedding": llm_embedding.cpu().numpy()  # Moving to CPU and converting to NumPy
        }

        # 获取 Stage 1 的输入名称并运行推理
        input_names_stage1 = [input.name for input in self.llm_stage1_session.get_inputs()]
        print(f"input_names_stage1:{input_names_stage1}")
        lm_input, text_len_output, speech_embedding_weight = self.llm_stage1_session.run(None, {
            input_names_stage1[0]: inputs_stage1["text"],
            input_names_stage1[1]: inputs_stage1["text_len_input"],
            input_names_stage1[2]: inputs_stage1["prompt_text"],
            input_names_stage1[3]: inputs_stage1["prompt_text_len"],
            input_names_stage1[4]: inputs_stage1["prompt_speech_token"],
            input_names_stage1[5]: inputs_stage1["embedding"]
        })
        
        #print(f"lm_input: {lm_input}")
        #print(f"text_len_output: {text_len_output}")

        # 初始化缓存
        att_cache = np.zeros((0, 0, 0, 128), dtype=np.float32) 
        #cnn_cache = np.zeros((0, 0, 0, 0), dtype=np.float32)

        # 计算最小和最大长度
        min_len = int((text_len_output - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len_output - prompt_text_len) * max_token_text_ratio)

        out_tokens = []
        offset = 0

        # Stage 2 的输入名称
        input_names_stage2 = [input.name for input in self.llm_stage2_session.get_inputs()]
        print(f"input_names_stage2: {input_names_stage2}")

        # 循环推理
        for i in range(max_len):
            #print(f"att_cache: {att_cache.shape}")
            #print(f"lm_input: {lm_input}")
            logp, att_cache, cnn_cache = self.llm_stage2_session.run(None, {
                input_names_stage2[0]: lm_input,
                input_names_stage2[1]: att_cache
            })
            
            # 采样生成 token
            #print(f"logp: {logp}")
            top_ids = self.sampling_ids(np.squeeze(logp, axis=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            #print(f"top_ids: {top_ids}")

            # 判断是否达到终止条件
            if top_ids == self.speech_token_size:
                break      

            out_tokens.append(top_ids)
            offset += lm_input.shape[1]
            lm_input = speech_embedding_weight[top_ids].reshape(1, 1, -1)
            #lm_input = speech_embedding.weight[top_ids].reshape(1, 1, -1).numpy()
            

        # 将生成的 tokens 转换为 numpy 数组并存储
        out_tokens_numpy = np.array(out_tokens, dtype=np.int64)
        out_tokens_unsqueezed = np.expand_dims(out_tokens_numpy, axis=0)

        for token in out_tokens_unsqueezed:
            self.tts_speech_token_dict[uuid].append(token)

        # 标记任务结束
        self.llm_end_dict[uuid] = True


    '''
    mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10
    '''
    def ODE_Solver(self, mu, mask, n_timesteps=10, temperature=1.0, inference_cfg_rate=0.7, embedding=None, conds=None):
        
        #print(f"temperature: {temperature}")
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        t, dt = t_span[0], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        x = z.clone()
        
        #print(f"conds: {conds.shape}")
        print(f"x: {x}")
        #print(f"mu: {mu.shape}")
        #print(f"embedding: {embedding.shape}")
        inputs = {
            'x': x.numpy(),
            'mask': mask.numpy(),
            'mu': mu.numpy(),     
            't': t.numpy(),
            'spks': embedding,
            'cond': conds.numpy()
        }
        #print(f"z: {z}")
        #print(f"mask: {mask}")
        #print(f"mu: {mu}")
        #print(f"t: {t}")
        #print(f"spks: {embedding}")
        #print(f"conds: {conds}")
        input_names = [input.name for input in self.flow_stage2_session.get_inputs()]
        for step in range(1, len(t_span)):
            dphi_dt = self.flow_stage2_session.run(None, {
                input_names[0]: inputs["x"],
                input_names[1]: inputs["mask"],
                input_names[2]: inputs["mu"],
                input_names[3]: inputs["t"],
                input_names[4]: inputs["spks"],
                input_names[5]: inputs["cond"]
            })
            if inference_cfg_rate > 0:
                cfg_dphi_dt = self.flow_stage2_session.run(None, {
                    input_names[0]: x.numpy(),
                    input_names[1]: mask.numpy(),
                    input_names[2]: torch.zeros_like(mu).numpy(),
                    input_names[3]: t.numpy(),
                    input_names[4]: np.zeros_like(embedding) if embedding is not None else None,
                    input_names[5]: torch.zeros_like(conds).numpy()
                })
                dphi_dt = ((1.0 + inference_cfg_rate) * dphi_dt[0] -
                           inference_cfg_rate * cfg_dphi_dt[0])
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        return x

    
    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False):

        #print(f"token: {token}")
        data = [667, 692, 658, 658, 658, 658, 632, 373, 685, 669, 700, 3965, 699, 383,
        427, 190, 223, 2437, 72, 596, 556, 603, 15, 3823, 620, 553, 553, 553,
        2828, 1006, 2828, 59, 223, 206, 179, 1307, 1307, 179, 1938, 260, 223, 1006,
        1006, 590, 2185, 471, 570, 28, 433, 75, 237, 237, 53, 707, 98, 418,
        714, 316, 126, 59, 414, 420, 393, 353, 207, 281, 676, 57, 115, 302,
        2386, 84, 533, 10, 3836, 570, 223, 1460, 946, 1104, 171, 420, 87, 626,
        412, 4, 318, 104, 1923, 340, 250, 374, 115, 115, 218, 1923, 187, 100,
        676, 514, 308, 383, 515, 726, 351, 11, 3946, 171, 304, 173, 476, 47,
        281, 308, 540, 35, 67, 1531, 2723, 2723, 2723, 56, 498, 277, 21, 415,
        563, 1381, 325, 325, 118, 51, 384, 83, 669, 669, 3190, 3190, 667, 667,
        667, 667, 667, 667, 667, 667, 667, 685, 659, 230, 216, 216, 447, 216,
        216, 216, 216, 216, 75, 345, 707, 192, 232, 38, 4030, 713, 1235, 261,
        526, 337, 98, 570, 453, 332, 25, 388, 348, 1104, 536, 33, 551, 422,
        498, 359, 728, 177, 3290, 7, 479, 187, 594, 273, 2031, 1531, 2280, 694,
        648, 160, 388, 41, 87, 406, 197, 597, 719, 4, 3889, 386, 718, 199,
        199, 199, 157, 475, 407, 28, 566, 230, 442, 216, 140, 230, 204, 85,
        3132, 468, 2723, 445, 56, 166, 212, 3766, 3965, 699, 363, 205, 574, 219,
        540, 393, 466, 515, 515, 308, 281, 240, 470, 44, 603, 645, 286, 596,
        726, 212, 103, 74, 325, 1381, 415, 126, 155, 10, 356, 576, 120, 21,
        265, 265, 92, 163, 554, 207, 1035, 436, 249, 331, 3190, 3190, 667, 667,
        3190, 658, 463, 531, 230, 230, 216, 216, 216, 210, 140, 204, 450, 95,
        514, 152, 3823, 568, 349, 523, 633, 314, 209, 254, 597, 4, 4, 648,
        2723, 157, 469, 1401, 460, 87, 591, 3766, 466, 94, 2437, 118, 579, 1596,
        406, 406, 3406, 536, 117, 144, 505, 75, 230, 216, 210, 140, 334, 3254,
        23, 1852, 3889, 318, 308, 3290, 62, 13, 1104, 434, 202, 3480, 188, 2057,
        358, 168, 448, 448, 484, 271, 332, 254, 166, 495, 33, 536, 607, 1354,
        47, 1930, 112, 285, 2579, 574, 187, 309, 100, 1938, 228, 338, 3823, 261,
        2644, 2644, 712, 434, 398, 331, 331, 3190, 619, 221, 667, 667, 667, 667,
        667, 667, 667, 667, 3190, 505, 2444, 71, 75, 216, 216, 237, 237, 216,
        216, 216, 216, 216, 216, 216, 75, 2444, 121, 3821, 602, 274, 3575, 3575,
        553, 217, 275, 507, 507, 2752, 407, 250, 527, 206, 114, 892, 441, 260,
        223, 547, 82, 4013, 4013, 433, 652, 631, 2444, 75, 53, 345, 85, 455,
        4, 4, 3889, 315, 318, 179, 275, 385, 97, 258, 892, 416, 79, 1228,
        110, 101, 424, 150, 423, 207, 165, 241, 371, 52, 362, 277, 92, 48,
        554, 2691, 2691, 662, 672, 529, 170, 202, 385, 612, 482, 42, 132, 257,
        15, 134, 16, 16, 16, 516, 135, 1214, 493, 270, 598, 696, 3190, 3190,
        692, 667, 667, 667, 667, 667, 667, 3190, 3821, 1091, 187, 742, 1, 2031,
        39, 2723, 468, 2514, 690, 52, 155, 10, 10, 623, 711, 2645]
        token = np.array(data, dtype=np.int64)
        #print(f"token: {token}")
        mel_len1 = prompt_feat.shape[1]
        token_len = np.array([token.shape[0]], dtype=np.int32)
        mel_len2 = int(token.shape[0] / 50 * 22050 / 256)
        prompt_token_len = np.array([prompt_token.shape[1]], dtype=np.int32)
        
        sum_token_len = token_len + prompt_token_len
        # print(f"sum_token_len: {sum_token_len}")  sum_token_len: [745]
        # 注意pytorch
        mask = (~make_pad_mask(torch.from_numpy(sum_token_len))).float().unsqueeze(-1)

        inputs = {
            'token': token,            
            'token_len': token_len,     
            'prompt_token': prompt_token.cpu().numpy(),  
            'prompt_token_len': prompt_token_len, 
            'prompt_feat': prompt_feat.cpu().numpy(),
            'embedding': embedding.cpu().numpy(),
            'mask': mask.cpu().numpy()
        }
        
        '''  for DEBUG
        model_dir = "E:\\Codes\\CosyVoice\\pretrained_models\\CosyVoice-300M"
        flow_model_stage1 = FlowModelStage1(model_dir=model_dir, device='cuda', verbose=True, max_batch_size=16, fp16=False)
        flow_model_s1 = flow_model_stage1.get_model()
        h, embedding, t_l = flow_model_s1(torch.from_numpy(token).cuda(), torch.from_numpy(token_len).cuda(), prompt_token.cuda(), 
                                     torch.from_numpy(prompt_token_len).cuda(), prompt_feat.cuda(), embedding.cuda(), mask.cuda())
        print(f"h: {h.shape}")
        print(f"embedding: {embedding.shape}")
        print(f"t_l: {t_l}")
        '''

        #print_variable_info(token=token, prompt_token=prompt_token, prompt_feat=prompt_feat, embedding=embedding)
        #print(f"token: {token.shape[0]}")
        #print(f"token: {prompt_token.shape}")

        # 获取输入名称并运行模型
        input_names = [input.name for input in self.flow_stage1_session.get_inputs()]
        print("input_names: {input_names}")
        h, embedding = self.flow_stage1_session.run(None, {
            input_names[0]: inputs["token"],
            input_names[1]: inputs["token_len"],
            input_names[2]: inputs["prompt_token"],
            input_names[3]: inputs["prompt_token_len"],
            input_names[4]: inputs["prompt_feat"],
            input_names[5]: inputs["embedding"],
            input_names[6]: inputs["mask"],
        })
        #print(f"h: {h}")
        
        #print(f"mel_len1: {mel_len1}")  mel_len1: 299 
        print(f"mel_len2: {mel_len2}") 
        conds = torch.zeros([1, mel_len1 + mel_len2, self.flow_output_size])
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).float()
        mask=mask.unsqueeze(1)
        print(f"prompt_token_len: {prompt_token_len}")
        print(f"h: {h.shape}")
        #print(f"t_l: {t_l}")
        mu=torch.from_numpy(h).transpose(1, 2).contiguous()
        feat = self.ODE_Solver(mu, mask, n_timesteps=10, temperature=1.0, inference_cfg_rate=0.7, embedding=embedding, conds=conds)
        tts_mel = feat[:, :, mel_len1:].to(self.device)
        assert tts_mel.shape[2] == mel_len2
        
        #print(f"tts_mel: {tts_mel}")
        
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            self.hift_cache_dict[uuid] = {'source': tts_source[:, :, -self.source_cache_len:], 'mel': tts_mel[:, :, -self.mel_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

        return tts_speech



    def inference(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
              prompt_text=torch.zeros(1, 0, dtype=torch.int32),
              llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
              flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
              prompt_speech_feat=torch.zeros(1, 0, 80), **kwargs):
    
        # 生成唯一ID，用于跟踪这次推理任务
        this_uuid = str(uuid.uuid1())
    
        # 初始化相关字典的条目
        self.tts_speech_token_dict[this_uuid] = []
        self.llm_end_dict[this_uuid] = False
        self.mel_overlap_dict[this_uuid] = None
        self.hift_cache_dict[this_uuid] = None
    
        # 执行 LLM 任务（非多线程）
        self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid)
        logging.info(f"LLM job finished for UUID: {this_uuid}")
    

        #print(f"self.tts_speech_token_dict[this_uuid]: {self.tts_speech_token_dict[this_uuid]}")
        # 处理所有生成的 tokens (非流模式)
        if self.tts_speech_token_dict[this_uuid]:
            # 使用 np.concatenate 来连接数组，假设这些数组都是 NumPy 数组
            this_tts_speech_token = np.concatenate(np.array(self.tts_speech_token_dict[this_uuid]), axis=0)
        else:
            logging.warning(f'tts_speech_token_dict[{this_uuid}] is empty. Returning a default value.')
            # 创建一个形状为 (1, 0, expected_token_size) 的零数组
            this_tts_speech_token = np.zeros((1, 0, self.expected_token_size))
    
        logging.info(f"Starting token to audio conversion for UUID: {this_uuid}")
    
        # 生成最终的语音输出
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            finalize=True
        )
    
        # 清理跟踪数据，移除对应的UUID条目
        for cache_dict in [self.tts_speech_token_dict, self.llm_end_dict, self.mel_overlap_dict, self.hift_cache_dict]:
            cache_dict.pop(this_uuid, None)
    
        # 返回生成的语音输出
        return {'tts_speech': this_tts_speech.cpu()}
    
    


import numpy as np
def print_variable_info(**kwargs):
    for name, var in kwargs.items():
        if isinstance(var, torch.Tensor):
            print(f"{name}: shape = {var.shape}, dtype = {var.dtype}")
        elif isinstance(var, np.ndarray):
            print(f"{name}: shape = {var.shape}, dtype = {var.dtype}")
        else:
            # 对于非张量/数组类型，打印类型而不是形状和dtype
            print(f"{name}: type = {type(var)}")


