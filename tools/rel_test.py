import torch, os
import numpy as np
from torch.utils.data import DataLoader
from datasets import PVSGRelationDataset
from models.relation_head.base import ObjectEncoder, PairProposalNetwork
from models.relation_head.base import VanillaModel
from models.relation_head.convolution import HandcraftedFilter, Learnable1DConv
from models.relation_head.transformer import TemporalTransformer
from models.relation_head.train_utils import concatenate_sub_obj
from models.relation_head.test_utils import pick_top_pairs_eval, \
    generate_results, generate_pairwise_results
from utils.rel_metrics import calculate_pair_recall_at_k, calculate_final_metrics, calculate_iou
from utils.show_log import save_metrics_to_csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the dataset and dataloader
data_dir = './data/'
split = 'val'
model_name = 'transformer' # vanilla, filter, conv, transformer
model_pth = 'epoch_100.pth'
mark = 'full_result'

mark = model_name + '_' + mark + '_' + model_pth.split('.')[0]
work_dir = f'./work_dirs/{split}_save_qf_1106'
save_work_dir = f'./work_dirs/rel_ips_{model_name}_100e_pe_ln'
csv_file_path = os.path.join(save_work_dir, 'full_result.csv')
loaded_state_dicts = torch.load(os.path.join(save_work_dir, model_pth))

pvsg_rel_dataset = PVSGRelationDataset(f"{data_dir}/pvsg.json", split, work_dir)
data_loader = DataLoader(pvsg_rel_dataset, batch_size=1, shuffle=False)
relation_list = pvsg_rel_dataset.relations

# for pairing
feature_dim = 256
hidden_dim = 1024

# for relation network
input_dim = 512

# for dataset
num_relations = 65
num_top_pairs = 100
max_frame_length = 900
accumulation_steps = 8

# load models
subject_encoder = ObjectEncoder(feature_dim=feature_dim)
subject_encoder.load_state_dict(loaded_state_dicts['subject_encoder'])
subject_encoder.to(device).eval()

object_encoder = ObjectEncoder(feature_dim=feature_dim)
object_encoder.load_state_dict(loaded_state_dicts['object_encoder'])
object_encoder.to(device).eval()

pair_proposal_model = PairProposalNetwork(feature_dim, hidden_dim)
pair_proposal_model.load_state_dict(loaded_state_dicts['pair_proposal_model'])
pair_proposal_model.to(device).eval()

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

relation_model.load_state_dict(loaded_state_dicts['relation_model'])
relation_model.to(device).eval()


def evaluate(subject_encoder, object_encoder, pair_proposal_model, relation_model, data_loader, num_top_pairs, relation_list, device, csv_file_path, mark):
    
    # for evaluation
    K_values = [20, 50, 100]
    relation_recall_dict = {K: {idx: {'name': name, 'total': 0, 'hit': 0, 'weak_hit': 0} \
        for idx, name in enumerate(relation_list)} for K in K_values}
        
    subject_encoder.eval()
    object_encoder.eval()
    pair_proposal_model.eval()
    relation_model.eval()

    # metrics
    pair_recall_list = []

    for i, relation_dict in enumerate(data_loader):
        with torch.no_grad():
            feats = relation_dict['feats'][0].float().to(device)
            gt_relations = relation_dict['relations']
            
            # Convert the features into subjects or objects
            sub_feats = subject_encoder(feats)
            obj_feats = object_encoder(feats)
            
            # Forward pass through the Pair Proposal Network
            pred_matrix = pair_proposal_model(sub_feats, obj_feats)
            
            # Get GT matrix for pair_matrix supervision    
            gt_pairs = [[int(relation['subject_index'].item()), int(relation['object_index'].item())] for relation in gt_relations]
            
            # get top pairs
            selected_pairs = pick_top_pairs_eval(pred_matrix, num_top_pairs)

            # compute recall@20
            pair_recall = calculate_pair_recall_at_k(selected_pairs, gt_pairs, 20)
            pair_recall_list.append(pair_recall)

            # evaluate the performance of pair selecting
            concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats, selected_pairs)
            span_pred, prob = relation_model(concatenated_feats)
            
            # the default strategy is generate_results, alternative is generate_pairwise_results
            # results = generate_results(span_pred, prob, selected_pairs)
            results = generate_pairwise_results(span_pred, prob, selected_pairs)


            for gt_relation in gt_relations:
                rel_key = (int(gt_relation['subject_index'].item()), 
                            int(gt_relation['object_index'].item()), 
                            int(gt_relation['relation'].item()))

                for K_value in K_values:
                    relation_recall_dict[K_value][rel_key[2]]['total'] += 1

                for idx, result in enumerate(results):
                    if (result['subject_index'], result['object_index'], result['relation']) == rel_key:
                        # print("A successful recall for", relation_list[rel_key[2]], flush=True)
                        t_iou = calculate_iou(gt_relation['relation_span'], result['relation_span'])
                        for K_value in K_values:
                            if idx < K_value:
                                relation_recall_dict[K_value][rel_key[2]]['weak_hit'] += 1
                                if t_iou >= 0.5:
                                    relation_recall_dict[K_value][rel_key[2]]['hit'] += 1
                        break

            torch.cuda.empty_cache()
            del concatenated_feats, span_pred, prob

    # final result
    print(f"Pair Recall@20: {100 * np.array(pair_recall_list).mean():.2f}")
    final_metrics = calculate_final_metrics(relation_recall_dict, K_values)
    for K in K_values:
        print('-------------------------------------------------------------------')
        print(f"Recall@{K}: {100 * final_metrics[K]['recall']:.2f}")
        print(f"Mean Recall@{K}: {100 * final_metrics[K]['mean_recall']:.2f}")
        print(f"Weak Recall@{K}: {100 * final_metrics[K]['weak_recall']:.2f}")
        print(f"Weak Mean Recall@{K}: {100 * final_metrics[K]['weak_mean_recall']:.2f}")
        print('-------------------------------------------------------------------')

    save_metrics_to_csv(final_metrics, pair_recall_list, K_values, csv_file_path, mark)

if __name__ == "__main__":
    evaluate(subject_encoder, object_encoder, pair_proposal_model, relation_model, data_loader, num_top_pairs, relation_list, device, csv_file_path, mark)
