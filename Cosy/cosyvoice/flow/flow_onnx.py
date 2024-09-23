import torch
import torch.nn as nn
from torch.nn import functional as F
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
                prompt_token,
                prompt_token_len,
                prompt_feat,
                embedding_input,
                mask):
        
        token = token.unsqueeze(0)
        # xvec projection
        embedding_output = F.normalize(embedding_input, dim=1)
        embedding_output = self.spk_embed_affine_layer(embedding_output)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        
        
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2)

        return h, embedding_output

    
    '''
    def forward(self,
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
        
        
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / 50 * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        

        # mask = (~make_pad_mask(feat_len)).to(h)
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)

        return h, mask, embedding, conds
        '''
    
    


