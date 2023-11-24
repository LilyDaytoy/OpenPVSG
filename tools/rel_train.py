import torch, os, argparse
import random, gc
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='prepare relation set')
parser.add_argument('--ps-type', default='vps', help='vps or ips output')
parser.add_argument('--model-name',
                    default='transformer',
                    help='vanilla, filter, conv, transformer')
args = parser.parse_args()

# Initialize the dataset and dataloader
data_dir = './data/'
# ps_type = 'vps'
# model_name = 'transformer' # vanilla, filter, conv, transformer
ps_type = args.ps_type
model_name = args.model_name
work_dir = f'./work_dirs/{ps_type}_train_save_qf'
work_dir_eval = f'./work_dirs/{ps_type}_save_qf'
save_work_dir = f'./work_dirs/relation/rel_{ps_type}_{model_name}_lr0.0001'
os.makedirs(save_work_dir, exist_ok=True)
mark = 'debug'
csv_file_path = os.path.join(save_work_dir, 'result.csv')

pvsg_rel_dataset = PVSGRelationDataset(f'{data_dir}/pvsg.json', 'train',
                                       work_dir)
data_loader = DataLoader(pvsg_rel_dataset, batch_size=1, shuffle=True)

pvsg_rel_dataset_eval = PVSGRelationDataset(f'{data_dir}/pvsg.json', 'val',
                                            work_dir_eval)
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
num_relations = 57
num_top_pairs = 50
num_max_samples = 100
max_frame_length = 900
accumulation_steps = 32

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
    relation_model = model_classes[model_name](input_dim,
                                               num_relations).to(device)
else:
    raise ValueError(f'Model {model_name} is unsupported')

# Define the optimizer
optimizer = optim.Adam(
    list(subject_encoder.parameters()) + list(object_encoder.parameters()) +
    list(pair_proposal_model.parameters()) + list(relation_model.parameters()),
    lr=0.0001,
)
# scheduler = CosineAnnealingLRwithWarmUp(optimizer, warmup_epochs=5, max_lr=0.001, min_lr=1e-7, num_epochs=num_epochs)

# compute relation numbers
relation_count = torch.ones(num_relations)
for i, relation_dict in enumerate(data_loader):
    gt_relations = relation_dict['relations']
    for gt_relation in gt_relations:
        relation_count[int(gt_relation['relation'].item())] += 1

print('Start Training', flush=True)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    for i, relation_dict in enumerate(data_loader):
        feats = relation_dict['feats'][0].float().to(device)
        gt_relations = relation_dict['relations']

        # cut feat sample size to avoid OOM
        if feats.size(0) > num_max_samples:
            unique_indices = set()
            for relation in gt_relations:
                unique_indices.add(relation['subject_index'].item())
                unique_indices.add(relation['object_index'].item())

            # Ensure unique_indices is within the max_samples limit
            if len(unique_indices) > num_max_samples:
                unique_indices = set(
                    random.sample(unique_indices, num_max_samples))

            # Fill the rest of selected_indices randomly if necessary
            remaining_slots = num_max_samples - len(unique_indices)
            all_indices = set(range(feats.size(0)))
            remaining_indices = all_indices - unique_indices
            selected_indices = list(unique_indices) + random.sample(
                remaining_indices, min(remaining_slots,
                                       len(remaining_indices)))

            # Sample feats based on selected_indices
            feats = feats[selected_indices, :, :]

            # Update the indices in gt_relations to match the sampled feats
            index_map = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(selected_indices)
            }
            updated_gt_relations = []
            for relation in gt_relations:
                if relation['subject_index'].item() in index_map and relation[
                        'object_index'].item() in index_map:
                    updated_gt_relations.append({
                        'subject_index':
                        torch.tensor(
                            [index_map[relation['subject_index'].item()]]),
                        'object_index':
                        torch.tensor(
                            [index_map[relation['object_index'].item()]]),
                        'relation':
                        relation['relation'],
                        'relation_span':
                        relation['relation_span'],
                    })
            gt_relations = updated_gt_relations

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
        loss_pair = zlpr_loss(gt_matrix.reshape(1, -1),
                              pred_matrix.reshape(1, -1).to(device))

        # get top pairs and concatenate features, but no need in training
        # selected_pairs = pick_top_pairs(gt_relations, pred_matrix, num_top_pairs)
        # get gt pairs
        # import pdb; pdb.set_trace();
        num_top_pairs = min(50, 10000 // obj_feats.shape[1])
        selected_pairs = get_gt_pairs(gt_relations, num_top_pairs)
        if selected_pairs == []: continue
        concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats,
                                                 selected_pairs)

        # run temporal models
        # print(relation_dict['vid'][0], concatenated_feats.shape, flush=True)
        span_pred, prob = relation_model(concatenated_feats)
        # print(relation_dict['vid'][0], concatenated_feats.shape, flush=True)
        # print(torch.cuda.memory_summary(device=None, abbreviated=True), flush=True)

        # Get GT content
        gt_span, gt_prob = generate_gt_matrix(gt_relations, selected_pairs,
                                              span_pred.shape, custom_span)

        # Compute loss for span_pred and prob
        loss_prob = rew_bce_loss(gt_prob.to(device), prob,
                                 relation_count.to(device))
        gt_span_filtered, span_pred_filtered = reshape_and_filter(
            gt_span.to(device), span_pred)
        loss_span_pred = zlpr_loss(gt_span_filtered, span_pred_filtered)

        loss = loss_pair + 5 * loss_prob + loss_span_pred
        loss = loss / accumulation_steps
        loss.backward()

        # Perform accumulating gradients
        if (i + 1) % accumulation_steps == 0 or i + 1 == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()
            print(
                f'Epoch: {epoch + 1}, Batch: {i + 1}, {loss_pair.item():.2f}|{loss_prob.item():.2f}|{loss_span_pred.item():.2f} Loss: {loss.item():.2f}',
                flush=True)

        torch.cuda.empty_cache()
        del feats, sub_feats, obj_feats, pred_matrix, gt_matrix, concatenated_feats
        del span_pred, prob, gt_span, gt_prob, gt_span_filtered, span_pred_filtered
        gc.collect()

    # save models
    model_state_dicts = {
        'subject_encoder': subject_encoder.state_dict(),
        'object_encoder': object_encoder.state_dict(),
        'pair_proposal_model': pair_proposal_model.state_dict(),
        'relation_model': relation_model.state_dict()
    }

    torch.save(model_state_dicts,
               os.path.join(save_work_dir, f'epoch_{epoch+1}.pth'))
    updated_mark = f'{model_name}_{mark}_epoch{epoch+1}'
    print('Evaluation Starts...', flush=True)
    evaluate(subject_encoder, object_encoder, pair_proposal_model,
             relation_model, data_loader_eval, num_top_pairs, relation_list,
             device, csv_file_path, updated_mark)
    subject_encoder.train()
    object_encoder.train()
    pair_proposal_model.train()
    relation_model.train()
    # scheduler.step()
