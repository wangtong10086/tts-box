# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
import threading
import time
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
import logging
import onnxruntime as ort
from typing import List
from cosyvoice.utils.common import ras_sampling_onnx

class CosyVoiceModel:

    def __init__(self,
                 llm_model_stage1_path: str,
                 llm_model_stage2_path: str,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'] 
        print("init llm_stage1_session")
        self.llm_stage1_session = ort.InferenceSession(llm_model_stage1_path, providers=self.providers)
        print("init llm_stage2_session")
        self.llm_stage2_session = ort.InferenceSession(llm_model_stage2_path, providers=self.providers)
        self.flow = flow
        self.hift = hift
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
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.flow_hift_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.hift_cache_dict = {}
        
        self.speech_token_size = 4096
        self.sampling = ras_sampling_onnx

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location="cuda")
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location="cuda")
        self.llm.llm = llm_llm
    
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

        self.llm_end_dict[uuid] = True
        
    
    def token2wav_without_stream(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False):
        with self.flow_hift_context:
            tts_mel = self.flow.inference_without_stream(token=token.to(self.device),
                                        token_len=torch.tensor([token.shape[0]], dtype=torch.int32).to(self.device),
                                        prompt_token=prompt_token.to(self.device),
                                        prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_feat=prompt_feat.to(self.device),
                                        embedding=embedding.to(self.device))
            #print(f"tts_mel: {tts_mel}")
            # mel overlap fade in out
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


    
    def inference_without_stream(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
              prompt_text=torch.zeros(1, 0, dtype=torch.int32),
              llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
              flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
              prompt_speech_feat=torch.zeros(1, 0, 80), **kwargs):
        # Generate a unique ID for tracking this inference
        this_uuid = str(uuid.uuid1())

        # Initialize tracking variables
        self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid], self.mel_overlap_dict[this_uuid], self.hift_cache_dict[this_uuid] = [], False, None, None

        # Directly run the LLM job (no threading)
        self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid)
        print("llm job over !!!")
        #print(f"self.tts_speech_token_dict[this_uuid]: {self.tts_speech_token_dict[this_uuid]}")

        # Process all tokens at once (non-stream mode)
        # 确保 tts_speech_token_dict[this_uuid] 是一个包含张量的列表
        #print(f"self.tts_speech_token_dict[this_uuid]: {self.tts_speech_token_dict[this_uuid]}")
        if len(self.tts_speech_token_dict[this_uuid]) > 0:
            this_tts_speech_token = torch.cat([torch.tensor(x) for x in self.tts_speech_token_dict[this_uuid]], dim=0)
            #this_tts_speech_token = torch.cat(self.tts_speech_token_dict[this_uuid], dim=0)
        else:
            logging.warning(f'tts_speech_token_dict[{this_uuid}] is empty. Returning a default value.')
            this_tts_speech_token = torch.zeros(1, 0, self.expected_token_size)  # 使用适当的尺寸和类型
        #print(f"self.tts_speech_token_dict[this_uuid]: {self.tts_speech_token_dict[this_uuid]}")
        print("start token to audio !!!")
        # 生成最终语音输出
        with self.flow_hift_context:
            this_tts_speech = self.token2wav_without_stream(token=this_tts_speech_token.to(self.device),
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)


        # Clean up and remove tracking data
        self.tts_speech_token_dict.pop(this_uuid)
        self.llm_end_dict.pop(this_uuid)
        self.mel_overlap_dict.pop(this_uuid)
        self.hift_cache_dict.pop(this_uuid)

        # Synchronize CUDA if necessary
        torch.cuda.synchronize()

        # Return the final speech output
        return {'tts_speech': this_tts_speech.cpu()}

