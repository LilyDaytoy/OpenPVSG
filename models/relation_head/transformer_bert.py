import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=512, num_relations=65, num_transformer_layers=3):
        super(TemporalTransformer, self).__init__()
        self.num_relations = num_relations
        self.chunk_size = 512
        self.overlap = 50

        config = BertConfig(
            hidden_size=input_dim,
            num_hidden_layers=num_transformer_layers,
            num_attention_heads=4,
            intermediate_size=512
        )

        self.bert = BertModel(config)

        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)

        self.span_head = nn.Linear(input_dim // 4, num_relations)
        self.pred_head = nn.Linear(input_dim // 4, num_relations)


    def forward(self, x):      
        if x.size(1) > self.chunk_size:
            sequence_output = self.process_in_chunks(x)
        else:
            sequence_output = self.bert(inputs_embeds=x).last_hidden_state
        
        # Span prediction for each frame
        sequence_output = F.relu(self.fc1(sequence_output))
        sequence_output = F.relu(self.fc2(sequence_output))

        span_pred = self.span_head(sequence_output)

        relation_pred = self.pred_head(sequence_output)
        relation_pred = torch.max(relation_pred, dim=1).values

        return span_pred, relation_pred
    
    def process_in_chunks(self, x):
        total_length = x.size(1)
        num_chunks = (total_length - self.overlap) // (self.chunk_size - self.overlap) + 1
        output_size = x.size(0), total_length, x.size(2)

        # Preallocate the tensor for reassembled output
        reassembled_output = torch.zeros(output_size, device=x.device)

        for i in range(num_chunks):
            start_idx = i * (self.chunk_size - self.overlap)
            end_idx = min(start_idx + self.chunk_size, total_length)
            chunk = x[:, start_idx:end_idx, :]
            chunk_output = self.bert(inputs_embeds=chunk).last_hidden_state

            # Directly accumulate the results in the reassembled_output tensor
            reassembled_output[:, start_idx:end_idx, :].add_(chunk_output)

            # Handle the overlapping part using in-place division
            if i > 0 and start_idx < total_length:
                overlap_end_idx = min(start_idx + self.overlap, total_length)
                reassembled_output[:, start_idx:overlap_end_idx, :].div_(2)

        return reassembled_output
