from collections import namedtuple

import torch
import torch.nn as nn
from transformers import BertModel, Wav2Vec2Model

from models.bert_adapter import BertAdapter
from models.wav2vec2model_adapter import Wav2Vec2AdapterModel


base_wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')


class MultiModalClassifierAdapter(nn.Module):
    
    def __init__(self, num_labels=7, dropout_prob=0.1, num_tasks=2, initialize_weights=False):
        super(MultiModalClassifierAdapter, self).__init__()

        self.bert = BertModel.from_pretrained('klue/bert-base')
        if initialize_weights:
            self.bert.init_weights()  # without pretraining
        
        for param in self.bert.parameters():
            param.requires_grad = False
        self.text_encoder = nn.ModuleList([BertAdapter(self.bert, adapter_size=64) for _ in range(num_tasks)])

        self.speech_encoder = Wav2Vec2AdapterModel.from_pretrained('facebook/wav2vec2-base-960h')
        if initialize_weights:
            self.speech_encoder.init_weights()  # without pretraining

        self.speech_projector = nn.ModuleList([nn.Linear(768, 256) for _ in range(num_tasks)])
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(768 + 256, num_labels) for _ in range(num_tasks)])


    def forward(self, speech_inputs, text_inputs, task_id):
        
        text_outputs = self.text_encoder[task_id](**text_inputs)
        text_hidden_states = text_outputs.pooler_output  # [CLS] hidden state
        # text_hidden_states.shape = (batch_size, 768)

        speech_outputs = self.speech_encoder(
            task_id=task_id,
            input_values=speech_inputs['input_values'],
            attention_mask=speech_inputs['attention_mask']
        )

        speech_hidden_states = speech_outputs.last_hidden_state
        # .extract_features : 512 dim
        # .last_hidden_state : 768 dim
        speech_hidden_states = self.speech_projector[task_id](speech_hidden_states)

        # mean_pooling (attention mask에서 1인 스텝들만 고려)
        padding_mask = base_wav2vec2._get_feature_vector_attention_mask(
            speech_hidden_states.shape[1], speech_inputs['attention_mask'], add_adapter=True)
        speech_hidden_states[~padding_mask] = 0.0
        speech_hidden_states = speech_hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        # speech_hidden_states.shape = (batch_size, 256)

        hidden_states = torch.cat([text_hidden_states, speech_hidden_states], dim=-1)
        hidden_states = self.dropout(hidden_states)
        output = self.classifier[task_id](hidden_states)
        # output.shape = (batch_size, num_labels)

        ClassifierOutput = namedtuple('ClassifierOutput', ['logits'])

        return ClassifierOutput(output)
    

class MultiModalClassifier(nn.Module):
    
    def __init__(self, num_labels=7, dropout_prob=0.1):
        super(MultiModalClassifier, self).__init__()

        self.text_encoder = BertModel.from_pretrained('klue/bert-base')
        self.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')

        self.speech_projector = nn.Linear(768, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(768 + 256, num_labels)


    def forward(self, speech_inputs, text_inputs):
        
        text_outputs = self.text_encoder(**text_inputs)
        text_hidden_states = text_outputs.pooler_output  # [CLS] hidden state
        # text_hidden_states.shape = (batch_size, 768)

        speech_outputs = self.speech_encoder(**speech_inputs)
        speech_hidden_states = speech_outputs.last_hidden_state
        # .extract_features : 512 dim
        # .last_hidden_state : 768 dim
        speech_hidden_states = self.speech_projector(speech_hidden_states)

        # mean_pooling (attention mask에서 1인 스텝들만 고려)
        padding_mask = self.speech_encoder._get_feature_vector_attention_mask(
            speech_hidden_states.shape[1], speech_inputs['attention_mask'])
        speech_hidden_states[~padding_mask] = 0.0
        speech_hidden_states = speech_hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        # speech_hidden_states.shape = (batch_size, 256)

        hidden_states = torch.cat([text_hidden_states, speech_hidden_states], dim=-1)
        hidden_states = self.dropout(hidden_states)
        output = self.classifier(hidden_states)
        # output.shape = (batch_size, num_labels)

        ClassifierOutput = namedtuple('ClassifierOutput', ['logits'])

        return ClassifierOutput(output)
