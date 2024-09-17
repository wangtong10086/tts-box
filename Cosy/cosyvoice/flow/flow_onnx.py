import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask


class MaskedDiffWithXvecStage1(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 vocab_size: int = 4096,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.length_regulator = length_regulator

    
    def forward(self,
                token,
                token_len,
                token_len1,
                token_len2,
                prompt_feat,
                mask):
        
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2)

        # get conditions
        #conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        #conds[:, :mel_len1] = prompt_feat
        #conds = conds.transpose(1, 2)
        
        # 创建一个形状为 [1, mel_len2, output_size]，所有元素为0的张量
        remaining_conds = torch.zeros([1, mel_len2, self.output_size], device=token.device)

        # 使用 torch.cat 来合并 prompt_feat 和 remaining_conds
        # 这里，我们首先沿着第二个维度（索引为1）拼接它们，因为我们想要在序列长度上进行拼接
        conds = torch.cat([prompt_feat, remaining_conds], dim=1)

        # 然后，进行转置操作
        conds = conds.transpose(1, 2)

        return h, conds

    
    
    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  embedding):
        
        token = token.unsqueeze(0)
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        
        h, conds = self.forward(token, token_len, token_len1, token_len2, prompt_feat, mask)
        return h, conds



class MaskedDiffWithXvecStage2(torch.nn.Module):
    def __init__(self,
                 decoder: torch.nn.Module = None):
        super().__init__()
        self.decoder = decoder

    
    def forward(self,
                h,
                mask,
                conds,
                mel_len1,
                embedding):
        
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        #assert feat.shape[2] == mel_len2
        return feat

    
    @torch.inference_mode()
    def inference(self,
                  mel_len1,
                  mel_len2,
                  h,
                  conds,
                  embedding):
        
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat = self.forward(h, mask, conds, mel_len1, embedding)
        assert feat.shape[2] == mel_len2
        return feat
    
    


