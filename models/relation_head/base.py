import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaModel(nn.Module):
    def __init__(self, input_dim, num_relations):
        super(VanillaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)

        self.span_head = nn.Linear(input_dim // 4, num_relations)
        self.pred_head = nn.Linear(input_dim // 4, num_relations)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        span_pred = self.span_head(x)

        relation_pred = self.pred_head(x)
        relation_pred = torch.max(relation_pred, dim=1).values

        return span_pred, relation_pred


class ObjectEncoder(nn.Module):
    def __init__(self,
                 feature_dim=256,
                 hidden_dim=512,
                 num_heads=8,
                 num_layers=2):
        super(ObjectEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class PairProposalNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(PairProposalNetwork, self).__init__()
        self.pair_ffn = nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim),
                                      nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, encoded_subjects, encoded_objects):
        sub_tokens = encoded_subjects.max(dim=1).values
        obj_tokens = encoded_objects.max(dim=1).values
        num_objects = obj_tokens.size(0)
        pair_matrix = torch.zeros(num_objects, num_objects)

        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    combined_features = torch.cat(
                        [sub_tokens[i], obj_tokens[j]], dim=-1)
                    pair_matrix[i, j] = self.pair_ffn(combined_features)

        return pair_matrix
