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
from utils.rel_metrics import calculate_pair_recall_at_k, calculate_final_metrics, calculate_viou
from utils.show_log import save_metrics_to_csv
from utils.relation_matching import PVSGRelationAnnotation, get_pred_mask_tubes_one_video

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the dataset and dataloader
data_dir = './data/'
split = 'val'
model_name = 'vanilla' # vanilla, filter, conv, transformer
model_pth = 'epoch_100.pth'
mark = 'full_result'

mark = model_name + '_' + mark + '_' + model_pth.split('.')[0]
work_dir = f'./work_dirs/{split}_save_qf_1106'
save_work_dir = f'./work_dirs/rel_ips_{model_name}_1114_100e'
csv_file_path = os.path.join(save_work_dir, 'full_result.csv')
loaded_state_dicts = torch.load(os.path.join(save_work_dir, model_pth))

pvsg_rel_dataset = PVSGRelationDataset(f"{data_dir}/pvsg.json", split, work_dir, return_mask=True)
data_loader = DataLoader(pvsg_rel_dataset, batch_size=1, shuffle=False)
object_names = pvsg_rel_dataset.classes
relation_list = pvsg_rel_dataset.relations

pvsg_ann_dataset = PVSGRelationAnnotation(f"{data_dir}/pvsg.json", split)

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

    subject_encoder.eval()
    object_encoder.eval()
    pair_proposal_model.eval()
    relation_model.eval()

    for i, relation_dict in enumerate(data_loader):
        with torch.no_grad():
            vid = relation_dict['vid'][0]
            feats = relation_dict['feats'][0].float().to(device)

            # Convert the features into subjects or objects
            sub_feats = subject_encoder(feats)
            obj_feats = object_encoder(feats)
            
            # Forward pass through the Pair Proposal Network
            pred_matrix = pair_proposal_model(sub_feats, obj_feats)
            
            # get top pairs
            selected_pairs = pick_top_pairs_eval(pred_matrix, num_top_pairs)

            # evaluate the performance of pair selecting
            concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats, selected_pairs)
            span_pred, prob = relation_model(concatenated_feats)
            
            # results = generate_results(span_pred, prob, selected_pairs)
            results = generate_pairwise_results(span_pred, prob, selected_pairs)

            idx2key = relation_dict['idx2key']
            for key in idx2key:
                idx2key[key] = idx2key[key].item()

            object_dict = {}
            for idx, mask_dict in enumerate(relation_dict['masks']):
                if len(mask_dict) > 0:
                    object_dict[idx] = object_names[int(mask_dict['cid'][0])]
            
            for idx, result in enumerate(results[:10]):
                s_id, o_id = result['subject_index'], result['object_index']
                if s_id in object_dict and o_id in object_dict:
                    s_name, o_name = object_dict[s_id], object_dict[o_id]
                    s_id_map, o_id_map = idx2key[s_id], idx2key[o_id]
                    relation_name = relation_list[result['relation']]
                    print(f'{s_name}-{s_id_map} {relation_name} {o_name}-{o_id_map}')
                    import pdb; pdb.set_trace();

            torch.cuda.empty_cache()
            del concatenated_feats, span_pred, prob


if __name__ == "__main__":
    evaluate(subject_encoder, object_encoder, pair_proposal_model, relation_model, data_loader, num_top_pairs, relation_list, device, csv_file_path, mark)
