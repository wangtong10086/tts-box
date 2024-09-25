import time
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import logging
from cosyvoice.onnx_infer.model_onnx import CosyVoiceModel

class CosyVoiceONNX:

    def __init__(self, model_dir):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        with open('{}/cosyvoice_onnx.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct, 
                                          configs['allowed_special'])
        # Load ONNX models using ONNX Runtime
        self.llm_stage1_model_path = '{}/llm_model_stage1.onnx'.format(model_dir)
        self.llm_stage2_model_path = '{}/llm_model_stage2.onnx'.format(model_dir)
        self.flow_stage1_model_path = '{}/flow_model_stage1.onnx'.format(model_dir)
        self.flow_stage2_model_path = '{}/flow_model_stage2.onnx'.format(model_dir)
        self.hift_model_diir = '{}'.format(model_dir)
        self.model = CosyVoiceModel(self.llm_stage1_model_path, self.llm_stage2_model_path, self.flow_stage1_model_path, self.flow_stage2_model_path, self.hift_model_diir)

    # The rest of the class methods remain unchanged except for the inference methods
    # Here is an example of how to modify the inference method to use ONNX models
    
    def inference(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)  

        # Initialize a list to store all model outputs
        all_model_outputs = []

        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))

            # Run inference in non-stream mode, accumulate all model outputs
            model_outputs = self.model.inference(**model_input)

            print(f"model_outputs: {model_outputs}")
            speech_len = model_outputs['tts_speech'].shape[1] / 22050
            logging.info('generated speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            all_model_outputs.append(model_outputs)  # Collect the model output

            start_time = time.time()
                

        # Return all accumulated model outputs
        return all_model_outputs
    
