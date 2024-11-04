from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import sys
import os
from cosyvoice.onnx_infer.pipeline import CosyVoiceONNX

#sys.path.append(os.path.abspath('third_party/Matcha-TTS'))

'''
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
# sft usage
print(cosyvoice.list_avaliable_spks())
# change stream=True for chunk stream inference
for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
    torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
'''

'''
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], 22050)
'''




# 初始化模型
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
func = cosyvoice.inference_zero_shot_without_stream

#cosyvoice = CosyVoiceONNX('model_convert/onnx')
#func = cosyvoice.inference

'''
# 加载 zero-shot 提示语音
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
# 使用非流模式执行推理
model_outputs = func(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    '希望你以后能够做的比我还好呦。',
    prompt_speech_16k
)
'''
# 加载 zero-shot 提示语音
prompt_speech_16k = load_wav('15.wav', 16000)
# 使用非流模式执行推理
model_outputs = func(
    '在面对挑战时，他展现了非凡的勇气。',
    '往生堂定时大酬宾，购一送一，购二送三，多购多得。',
    prompt_speech_16k
)

# 保存生成的语音
for i, model_output in enumerate(model_outputs):
    torchaudio.save('zero_shot_{}.wav'.format(i), model_output['tts_speech'], 22050)



'''
# cross_lingual usage
prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k, stream=False)):
    torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], 22050)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.', stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], 22050)
'''