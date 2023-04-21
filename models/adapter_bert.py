from collections import namedtuple

import torch
import torch.nn as nn


class Adapter(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, std=1e-3):
        super().__init__()
        self.down_project = nn.Linear(input_dim, hidden_dim)
        self.up_project = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()
        
        for param in self.down_project.parameters():
            nn.init.trunc_normal_(param.data, std=std)
        for param in self.up_project.parameters():
            nn.init.trunc_normal_(param.data, std=std)
        
    def forward(self, x):
        y = self.down_project(x)
        y = self.gelu(y)
        y = self.up_project(y)
        return x + y
    

class BertAdapter(nn.Module):
    
    def __init__(self, bert_model, adapter_size=64):
        super().__init__()
        self.bert = bert_model
        config = bert_model.config
        self.mha_layernorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(12)])
        self.ffn_layernorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(12)])
        
        for i, layer_module in enumerate(self.bert.encoder.layer):
            self.mha_layernorm[i].load_state_dict(layer_module.attention.output.LayerNorm.state_dict())
            self.ffn_layernorm[i].load_state_dict(layer_module.output.LayerNorm.state_dict())
        
        self.mha_adapters = nn.ModuleList([Adapter(input_dim=768, hidden_dim=adapter_size) for _ in range(12)])
        self.ffn_adapters = nn.ModuleList([Adapter(input_dim=768, hidden_dim=adapter_size) for _ in range(12)])
    
    def forward(self, input_ids, attention_mask):
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape)
        
        hidden_states = self.bert.embeddings(input_ids)
        
        for i, layer_module in enumerate(self.bert.encoder.layer):
            
            # self_attention_outputs = layer_module.attention(
            #     hidden_states,
            #     extended_attention_mask
            # )
            self_outputs = layer_module.attention.self(
                hidden_states,
                extended_attention_mask
            )  # MHA
            output_hidden_states = layer_module.attention.output.dense(self_outputs[0])
            output_hidden_states = layer_module.attention.output.dropout(output_hidden_states)
            adapter_output = self.mha_adapters[i](output_hidden_states)
            attention_output = self.mha_layernorm[i](adapter_output + hidden_states)
            
            # layer_output = apply_chunking_to_forward(
            #     layer_module.feed_forward_chunk, layer_module.chunk_size_feed_forward, layer_module.seq_len_dim, attention_output
            # )
            intermediate_output = layer_module.intermediate(attention_output)  # FFN
            output_hidden_states = layer_module.output.dense(intermediate_output)
            output_hidden_states = layer_module.output.dropout(output_hidden_states)
            adapter_output = self.ffn_adapters[i](output_hidden_states)
            layer_output = self.ffn_layernorm[i](adapter_output + attention_output)
            
            hidden_states = layer_output
        
        pooled_output = self.bert.pooler(hidden_states)
        
        BertOutput = namedtuple('BertOutput', ['pooler_output'])
        
        return BertOutput(pooled_output)
