from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2Encoder,
    Wav2Vec2Adapter,
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    Wav2Vec2BaseModelOutput,
)

    
# class WavAdapter(nn.Module):
#     def __init__(self, in_size=768, hidden_size=256):
#         super().__init__()
#         self.down_project = nn.Linear(in_size, hidden_size)
#         self.nonlinearity = nn.GELU()
#         self.up_project = nn.Linear(hidden_size, in_size)
        
#     def forward(self, x):
#         x = self.down_project(x)
#         x = self.nonlinearity(x)
#         x = self.up_project(x)
#         return x

class MyWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config, n_tasks=2, freeze_encoder=True):
        '''
        args
            config
            n_tasks (int) the number of tasks
        '''
        print('initializing MyWav2Vec2Model...')
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        
        if freeze_encoder:
            print('freeze pretrained part of wav2vec')
            l = 0
            for n, p in self.feature_extractor.named_parameters():
                p.requires_grad = False
                l += 1
            print('feature_extractor:', l, 'are frozen')
            l = 0
            for n, p in self.feature_projection.named_parameters():
                p.requires_grad = False
                l += 1
            print('feature_projection:', l, 'are frozen')
        
#         # 우리 아답터
#         self.my_adapter = [WavAdapter() for _ in range(n_tasks)]
        
        if config.mask_time_prob > .0 or config.mask_feature_prob > .0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_(), requires_grad=False)
            
        if config.do_stable_layer_norm:
            print('config.do_stable_layer_norm', config.do_stable_layer_norm)
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)
        
        for n, p in self.encoder.named_parameters():
            p.requires_grad = False
        
        # An Adapter Based Pre-Training for Efficient and Scalable Self-Supervised Speech Representation Learning
#         self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None
        self.adapter = nn.ModuleList([Wav2Vec2Adapter(config) for _ in range(n_tasks)])
            
        self.post_init()
        
    def forward(
        self,
        task_id, # 필수
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)      
        
        extract_features = extract_features.transpose(1, 2)
    
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        
#         if len(self.my_adapter) != 0:
#             hidden_states = self.my_adapter[task_id](hidden_states)
# #             print('hidden_states', hidden_states.size()) # (bsz, 249, 768)
  
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = encoder_outputs[0]
        
        
        if self.adapter is not None:
            hidden_states = self.adapter[task_id](hidden_states)

        if not return_dict:
            return (hidden_states, extract_feature) + encoder_outputs[1:]
        
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# wav2vec2 = MyWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", **{"add_adapter":False})
# s, t, e = next(iter(train19_dl))
# input_values, attention_mask = s.get('input_values'), s.get('attention_mask')
# o = wav2vec2(
#     task_id = 0,
#     input_values = input_values,
#     attention_mask = attention_mask
# )