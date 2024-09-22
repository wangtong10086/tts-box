# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy


class TransformerLMStage1(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)


    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens


    def forward(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:  
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len = text_len + prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if torch.sum(embedding.shape[0] != 0) > 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        
        # 如果是SFT，也要走一次speech_embedding计算，只不过prompt_speech_token是一个全0张量
        prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        
        #if prompt_speech_token_len != 0:
        #    prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        #else:
        #    prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
        
        speech_embedding_weight = self.speech_embedding.weight

        return lm_input, text_len, speech_embedding_weight



class TransformerLMStage2(torch.nn.Module):
    def __init__(
            self,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
    ):
        super().__init__()
        self.speech_token_size = speech_token_size
    
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)


    def forward(
            self, 
            lm_input, 
            att_cache, 
            cnn_cache,
    ) -> torch.Tensor:  
        
        y_pred, att_cache_out, cnn_cache_out = self.llm.forward_chunk(lm_input, offset=0, required_cache_size=-1, att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool))
        logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        return logp, att_cache_out, cnn_cache_out
    
    
    



    

    