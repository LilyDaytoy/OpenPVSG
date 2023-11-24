import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    def __init__(self,
                 input_dim=512,
                 num_relations=57,
                 num_transformer_layers=1,
                 dropout_rate=0.1):
        super(TemporalTransformer, self).__init__()
        self.num_relations = num_relations
        self.positional_encoding = PositionalEncoding(input_dim,
                                                      dropout=dropout_rate)

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim,
                                                    nhead=4,
                                                    dim_feedforward=512,
                                                    dropout=dropout_rate)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_transformer_layers)
        self.layer_norm = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)

        self.span_head = nn.Linear(input_dim // 4, num_relations)
        self.pred_head = nn.Linear(input_dim // 4, num_relations)

    def forward(self, x):
        # Assuming x is of shape [batch_size, seq_length, input_dim]
        # Transformer expects input of shape [seq_length, batch_size, input_dim]
        x = x.transpose(0, 1)  # Swap batch and sequence length dimensions

        # Pass through Transformer
        x = self.positional_encoding(x)
        sequence_output = self.transformer_encoder(x)
        sequence_output = self.layer_norm(sequence_output)

        # Revert to original shape [batch_size, seq_length, input_dim]
        sequence_output = sequence_output.transpose(0, 1)

        # Span prediction for each frame
        sequence_output = F.relu(self.fc1(sequence_output))
        sequence_output = F.relu(self.fc2(sequence_output))

        span_pred = self.span_head(sequence_output)
        relation_pred = self.pred_head(sequence_output)
        relation_pred = torch.max(relation_pred, dim=1).values

        return span_pred, relation_pred


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
