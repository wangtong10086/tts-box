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

class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
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

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.llm.half()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location="cuda")
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location="cuda")
        self.llm.llm = llm_llm

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            for i in self.llm.inference(text=text.to(self.device),
                                                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                embedding=llm_embedding.to(self.device).half(),
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def llm_job_without_stream(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            for i in self.llm.inference_without_stream(text=text.to(self.device),
                                                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                embedding=llm_embedding.to(self.device).half(),
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False):
        with self.flow_hift_context:
            tts_mel = self.flow.inference(token=token.to(self.device),
                                        token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_token=prompt_token.to(self.device),
                                        prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_feat=prompt_feat.to(self.device),
                                        prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=embedding.to(self.device))
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

    
    def token2wav_without_stream(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False):
        with self.flow_hift_context:
            '''
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
            token = torch.tensor(data, dtype=torch.int32)
            '''
            #print(f"token: {token}")
            #print(f"token: {token}")
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

    def inference(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid], self.mel_overlap_dict[this_uuid], self.hift_cache_dict[this_uuid] = [], False, None, None
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len], dim=1)
                    with self.flow_hift_context:
                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                    prompt_token=flow_prompt_speech_token,
                                                    prompt_feat=prompt_speech_feat,
                                                    embedding=flow_embedding,
                                                    uuid=this_uuid,
                                                    finalize=False)
                    yield  {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid], dim=1)
            with self.flow_hift_context:
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            uuid=this_uuid,
                                            finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid], dim=1)
            with self.flow_hift_context:
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            uuid=this_uuid,
                                            finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        torch.cuda.synchronize()


    
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
        self.llm_job_without_stream(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid)
        print("llm job over !!!")
        #print(f"self.tts_speech_token_dict[this_uuid]: {self.tts_speech_token_dict[this_uuid]}")

        # Process all tokens at once (non-stream mode)
        # 确保 tts_speech_token_dict[this_uuid] 是一个包含张量的列表
        #print(f"self.tts_speech_token_dict[this_uuid]: {self.tts_speech_token_dict[this_uuid]}")
        if len(self.tts_speech_token_dict[this_uuid]) > 0:
            this_tts_speech_token = torch.cat(self.tts_speech_token_dict[this_uuid], dim=0)
        else:
            logging.warning(f'tts_speech_token_dict[{this_uuid}] is empty. Returning a default value.')
            this_tts_speech_token = torch.zeros(1, 0, self.expected_token_size)  # 使用适当的尺寸和类型

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

