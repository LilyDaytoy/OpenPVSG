from re import T
import torch, os
import random
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import PVSGRelationDataset
from models.relation_head.base import ObjectEncoder, PairProposalNetwork
from models.relation_head.base import VanillaModel
from models.relation_head.convolution import HandcraftedFilter, Learnable1DConv
from models.relation_head.transformer import TemporalTransformer
from models.relation_head.train_utils import *
from tools.rel_test import evaluate
from utils.lr_scheduler import CosineAnnealingLRwithWarmUp

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the dataset and dataloader
data_dir = './data/'
model_name = 'transformer' # vanilla, filter, conv, transformer
work_dir = f'./work_dirs/train_save_qf_1106'
work_dir_eval = f'./work_dirs/val_save_qf_1106'
save_work_dir = f'./work_dirs/rel_ips_{model_name}'
os.makedirs(save_work_dir, exist_ok=True)
mark = 'auto'
csv_file_path = os.path.join(save_work_dir, 'result.csv')

pvsg_rel_dataset = PVSGRelationDataset(f"{data_dir}/pvsg.json", 'train', work_dir)
data_loader = DataLoader(pvsg_rel_dataset, batch_size=1, shuffle=True)

pvsg_rel_dataset_eval = PVSGRelationDataset(f"{data_dir}/pvsg.json", 'val', work_dir_eval)
data_loader_eval = DataLoader(pvsg_rel_dataset, batch_size=1, shuffle=False)
relation_list = pvsg_rel_dataset_eval.relations

# Set the number of epochs
num_epochs = 100

# for pairing
feature_dim = 256
hidden_dim = 1024

# for relation network
input_dim = 512

# for dataset
num_relations = 65
num_top_pairs = 100
max_frame_length = 900
accumulation_steps = 64

subject_encoder = ObjectEncoder(feature_dim=feature_dim).to(device)
object_encoder = ObjectEncoder(feature_dim=feature_dim).to(device)

pair_proposal_model = PairProposalNetwork(feature_dim, hidden_dim).to(device)

assert model_name in ['vanilla', 'filter', 'conv', 'transformer'], \
    f'Model {model_name} unsupported'

model_classes = {
    'vanilla': VanillaModel,
    'filter': HandcraftedFilter,
    'conv': Learnable1DConv,
    'transformer': TemporalTransformer
}
if model_name in model_classes:
    relation_model = model_classes[model_name](input_dim, num_relations).to(device)
else:
    raise ValueError(f'Model {model_name} is unsupported')

# Define the optimizer
optimizer = optim.Adam(
    list(subject_encoder.parameters()) + list(object_encoder.parameters()) + 
    list(pair_proposal_model.parameters()) + 
    list(relation_model.parameters()), 
    lr=0.001,
)
# scheduler = CosineAnnealingLRwithWarmUp(optimizer, warmup_epochs=5, max_lr=0.001, min_lr=1e-7, num_epochs=num_epochs)

# compute relation numbers
# relation_count = torch.ones(num_relations)
# for i, relation_dict in enumerate(data_loader):
#     gt_relations = relation_dict['relations']
#     for gt_relation in gt_relations:
#         relation_count[int(gt_relation['relation'].item())] += 1

relation_count = torch.Tensor([131, 25, 56, 28, 37, 31, 11, 13, 32, 6, 41, 4, 18, 3, 15, 7, 4, 9, 1, 1, 1, 5, 7, 5, 851, 1, 20, 42, 275, 4, 2, 10, 31, 7, 3, 134, 62, 79, 802, 84, 5, 39, 36, 116, 7, 6, 20, 1, 1, 60, 11, 225, 6, 1, 199, 5, 18, 20, 32, 28, 130, 9, 126, 5, 30])

print('Start Training', flush=True)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    for i, relation_dict in enumerate(data_loader):
        feats = relation_dict['feats'][0].float().to(device)
        gt_relations = relation_dict['relations']

        # Randomly sample a chunk to avoid OOM
        if feats.size(1) > max_frame_length:
            start_index = random.randint(0, feats.size(1) - max_frame_length)
            custom_span = [start_index, start_index + max_frame_length]
        else:
            custom_span = [0, feats.size(1)]
        feats = feats[:, custom_span[0]:custom_span[1], :]
        
        # Convert the features into subjects or objects
        sub_feats = subject_encoder(feats)
        obj_feats = object_encoder(feats)
        
        # Forward pass through the Pair Proposal Network
        pred_matrix = pair_proposal_model(sub_feats, obj_feats)
        
        # Get GT matrix for pair_matrix supervision
        gt_matrix = torch.zeros_like(pred_matrix).to(device)
        for relation in gt_relations:
            gt_matrix[relation['subject_index'], relation['object_index']] = 1
        
        # Loss function for 
        loss_pair = zlpr_loss(gt_matrix.reshape(1,-1), pred_matrix.reshape(1,-1).to(device))
        
        # get top pairs and concatenate features
        # selected_pairs = pick_top_pairs(gt_relations, pred_matrix, num_top_pairs)
        # get gt pairs
        selected_pairs = get_gt_pairs(gt_relations)
        if selected_pairs == []: continue
        concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats, selected_pairs)

        # run temporal models
        span_pred, prob = relation_model(concatenated_feats)
        
        # Get GT content
        gt_span, gt_prob = generate_gt_matrix(gt_relations, selected_pairs, span_pred.shape, custom_span)
        
        # Compute loss for span_pred and prob
        loss_prob = rew_bce_loss(gt_prob.to(device), prob, relation_count.to(device))
        gt_span_filtered, span_pred_filtered = reshape_and_filter(gt_span.to(device), span_pred)        
        loss_span_pred = zlpr_loss(gt_span_filtered, span_pred_filtered)
        
        loss = loss_pair + 5 * loss_prob + loss_span_pred
        loss = loss / accumulation_steps
        loss.backward()

        # Perform accumulating gradients
        if (i + 1) % accumulation_steps == 0 or i + 1 == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, {loss_pair.item():.2f}|{loss_prob.item():.2f}|{loss_span_pred.item():.2f} Loss: {loss.item():.2f}', flush=True)
        
        torch.cuda.empty_cache()
        del concatenated_feats, span_pred, prob
    
    # save models
    model_state_dicts = {
        'subject_encoder': subject_encoder.state_dict(),
        'object_encoder': object_encoder.state_dict(),
        'pair_proposal_model': pair_proposal_model.state_dict(),
        'relation_model': relation_model.state_dict()
        }

    torch.save(model_state_dicts, os.path.join(save_work_dir, f'epoch_{epoch+1}.pth'))
    updated_mark=f'{model_name}_{mark}_epoch{epoch+1}'
    print('Evaluation Starts...', flush=True)
    evaluate(subject_encoder, object_encoder, pair_proposal_model, relation_model, data_loader_eval, num_top_pairs, relation_list, device, csv_file_path, updated_mark)
    subject_encoder.train()
    object_encoder.train()
    pair_proposal_model.train()
    relation_model.train()
    # scheduler.step()
