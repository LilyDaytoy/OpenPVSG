import torch
import torch.nn as nn
import torch.nn.functional as F


class HandcraftedFilter(nn.Module):
    def __init__(self, feat_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations

        self.filter_weights = torch.tensor([1 / 4, 1 / 2, 1, 1 / 2, 1 / 4],
                                           dtype=torch.float32)
        self.expanded_filter_weights = self.filter_weights.view(
            1, 1, -1).repeat(feat_dim, 1, 1)

        self.fc1 = nn.Linear(feat_dim, feat_dim // 2)
        self.fc2 = nn.Linear(feat_dim // 2, feat_dim // 4)

        self.span_head = nn.Linear(feat_dim // 4, num_relations)
        self.pred_head = nn.Linear(feat_dim // 4, num_relations)

    def forward(self, x):
        _, _, feat_dim = x.shape
        x = x.permute(0, 2, 1)
        filtered_x = F.conv1d(x,
                              self.expanded_filter_weights.to(x.device),
                              padding=2,
                              groups=feat_dim)
        filtered_x = filtered_x.permute(0, 2, 1)

        filtered_x = F.relu(self.fc1(filtered_x))
        filtered_x = F.relu(self.fc2(filtered_x))

        span_pred = self.span_head(filtered_x)

        relation_pred = self.pred_head(filtered_x)
        relation_pred = torch.max(relation_pred, dim=1).values

        return span_pred, relation_pred


class Learnable1DConv(nn.Module):
    def __init__(self, input_dim, num_relations, kernel_size=5, num_layers=1):
        super(Learnable1DConv, self).__init__()
        self.num_relations = num_relations

        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(input_dim,
                          input_dim,
                          kernel_size,
                          padding=kernel_size // 2))
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)

        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)

        self.span_head = nn.Linear(input_dim // 4, num_relations)
        self.pred_head = nn.Linear(input_dim // 4, num_relations)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_output = self.conv_layers(x)
        filtered_x = conv_output.permute(0, 2, 1)

        filtered_x = F.relu(self.fc1(filtered_x))
        filtered_x = F.relu(self.fc2(filtered_x))
        span_pred = self.span_head(filtered_x)

        relation_pred = self.pred_head(filtered_x)
        relation_pred = torch.max(relation_pred, dim=1).values

        return span_pred, relation_pred
