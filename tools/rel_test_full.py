import torch, os, argparse
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='prepare relation set')
parser.add_argument('--work-dir', help='vanilla, filter, conv, transformer')
parser.add_argument('--epoch-id', type=int, default='100')
args = parser.parse_args()

# Initialize the dataset and dataloader
data_dir = './data/'
split = 'val'
# model_name = 'vanilla' # vanilla, filter, conv, transformer
# model_pth = 'epoch_100.pth'
save_work_dir = args.work_dir
dir_name = save_work_dir.split('/')[-1]
ps_type = dir_name.split('_')[1]
model_name = dir_name.split('_')[2]

model_pth = f'epoch_{args.epoch_id}.pth'
mark = f'{dir_name}_standard'

mark = model_name + '_' + mark + '_' + model_pth.split('.')[0]
work_dir = f'./work_dirs/{ps_type}_{split}_save_qf'
# save_work_dir = f'./work_dirs/relations/rel_{ps_type}_{model_name}'
# csv_file_path = os.path.join(save_work_dir, 'full_result.csv')
csv_file_path = './work_dirs/relation/main_results.csv'
loaded_state_dicts = torch.load(os.path.join(save_work_dir, model_pth))

pvsg_rel_dataset = PVSGRelationDataset(f'{data_dir}/pvsg.json',
                                       split,
                                       work_dir,
                                       return_mask=True)
data_loader = DataLoader(pvsg_rel_dataset, batch_size=1, shuffle=False)
relation_list = pvsg_rel_dataset.relations

pvsg_ann_dataset = PVSGRelationAnnotation(f'{data_dir}/pvsg.json', split)

# for pairing
feature_dim = 256
hidden_dim = 1024

# for relation network
input_dim = 512

# for dataset
num_relations = 57
num_top_pairs = 100
max_frame_length = 900

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
    relation_model = model_classes[model_name](input_dim,
                                               num_relations).to(device)
else:
    raise ValueError(f'Model {model_name} is unsupported')

relation_model.load_state_dict(loaded_state_dicts['relation_model'])
relation_model.to(device).eval()


def evaluate(subject_encoder, object_encoder, pair_proposal_model,
             relation_model, data_loader, num_top_pairs, relation_list, device,
             csv_file_path, mark):

    # for evaluation
    K_values = [20, 50, 100]
    relation_recall_dict = {K: {idx: {'name': name, 'total': 0, 'hit': 0, 'weak_hit': 0} \
        for idx, name in enumerate(relation_list)} for K in K_values}
    soft_relation_recall_dict = {K: {idx: {'name': name, 'total': 0, 'hit': 0, 'weak_hit': 0} \
        for idx, name in enumerate(relation_list)} for K in K_values}

    subject_encoder.eval()
    object_encoder.eval()
    pair_proposal_model.eval()
    relation_model.eval()

    # metrics
    pair_recall_list = []

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
            concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats,
                                                     selected_pairs)
            span_pred, prob = relation_model(concatenated_feats)

            results = generate_results(span_pred, prob, selected_pairs)
            # results = generate_pairwise_results(span_pred, prob, selected_pairs)
            gt_dict = pvsg_ann_dataset[vid]

            # get object_id to category list
            gt_object_dict = {}
            for object_dict in gt_dict['objects']:
                gt_object_dict[
                    object_dict['object_id']] = object_dict['category']

            pred_obj_dict_with_mask = {}
            for idx, mask_dict in enumerate(relation_dict['masks']):
                pred_obj_dict_with_mask[idx] = mask_dict

            for gt_relation in gt_dict['relations']:
                # first check the triplets
                weak_hit_flag = 0
                sub_idx, obj_idx, rel_idx, gt_span_list = gt_relation
                rel_key = (int(gt_object_dict[sub_idx]),
                           int(gt_object_dict[obj_idx]), int(rel_idx))

                for K_value in K_values:
                    relation_recall_dict[K_value][rel_key[2]]['total'] += 1
                    soft_relation_recall_dict[K_value][
                        rel_key[2]]['total'] += 1

                for idx, result in enumerate(results):
                    if len(pred_obj_dict_with_mask[result['subject_index']]) == 0 \
                        or len(pred_obj_dict_with_mask[result['object_index']]) == 0:
                        continue

                    # import pdb; pdb.set_trace();
                    if (int(pred_obj_dict_with_mask[result['subject_index']]
                            ['cid'][0]),
                            int(pred_obj_dict_with_mask[result['object_index']]
                                ['cid'][0]), result['relation']) == rel_key:
                        # iou is the standard protocol, iou_weak ignores time span evaluation
                        iou, iou_weak = calculate_viou(
                            (sub_idx, obj_idx, gt_span_list),
                            (pred_obj_dict_with_mask[result['subject_index']]
                             ['mask'], pred_obj_dict_with_mask[
                                 result['object_index']]['mask'],
                             result['relation_span']), vid, data_dir)

                        # if weak_hit_flag == 0 and iou_weak >= 0.5:
                        #     weak_hit_flag = 1
                        #     for K_value in K_values:
                        #         if idx < K_value:
                        #             relation_recall_dict[K_value][rel_key[2]]['weak_hit'] += 1
                        #             soft_relation_recall_dict[K_value][rel_key[2]]['weak_hit'] += iou_weak

                        if iou >= 0.1:
                            for K_value in K_values:
                                if idx < K_value:
                                    relation_recall_dict[K_value][
                                        rel_key[2]]['weak_hit'] += 1
                                    soft_relation_recall_dict[K_value][
                                        rel_key[2]]['weak_hit'] += iou

                        if iou >= 0.5:
                            print(vid,
                                  relation_list[rel_key[2]],
                                  f'{100*iou:.2f}',
                                  f'{100*iou_weak:.2f}',
                                  flush=True)
                            for K_value in K_values:
                                if idx < K_value:
                                    relation_recall_dict[K_value][
                                        rel_key[2]]['hit'] += 1
                                    soft_relation_recall_dict[K_value][
                                        rel_key[2]]['hit'] += iou
                            break

            torch.cuda.empty_cache()
            del concatenated_feats, span_pred, prob

    # final result
    final_metrics = calculate_final_metrics(relation_recall_dict, K_values)
    final_soft_metrics = calculate_final_metrics(soft_relation_recall_dict,
                                                 K_values)
    for K in K_values:
        print(
            '-------------------------------------------------------------------'
        )
        print(f"Recall@{K}: {100 * final_metrics[K]['recall']:.2f}")
        print(f"Mean Recall@{K}: {100 * final_metrics[K]['mean_recall']:.2f}")
        print(f"Weak Recall@{K}: {100 * final_metrics[K]['weak_recall']:.2f}")
        print(
            f"Weak Mean Recall@{K}: {100 * final_metrics[K]['weak_mean_recall']:.2f}"
        )
        print('------------------------------------')
        print(f"Soft Recall@{K}: {100 * final_soft_metrics[K]['recall']:.2f}")
        print(
            f"Soft Mean Recall@{K}: {100 * final_soft_metrics[K]['mean_recall']:.2f}"
        )
        print(
            f"Soft Weak Recall@{K}: {100 * final_soft_metrics[K]['weak_recall']:.2f}"
        )
        print(
            f"Soft Weak Mean Recall@{K}: {100 * final_soft_metrics[K]['weak_mean_recall']:.2f}"
        )
        print(
            '-------------------------------------------------------------------'
        )

    save_metrics_to_csv(final_metrics, pair_recall_list, K_values,
                        csv_file_path, mark)


if __name__ == '__main__':
    evaluate(subject_encoder, object_encoder, pair_proposal_model,
             relation_model, data_loader, num_top_pairs, relation_list, device,
             csv_file_path, mark)
