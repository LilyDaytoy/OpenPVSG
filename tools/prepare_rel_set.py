import sys, time, argparse
from utils.relation_matching import *
sys.path.append('/mnt/lustre/jkyang/CVPR23/openpvsg')

parser = argparse.ArgumentParser(description='prepare relation set')
parser.add_argument('--data_dir',
                    default='./data',
                    help='path to pvsg dataset')
parser.add_argument('--work_dir', help='output result file in pickle format')
parser.add_argument('--split', help='generate train or val set')
args = parser.parse_args()

data_dir = args.data_dir
split = args.split
work_dir = args.work_dir

# get pvsg dataset
pvsg_dataset = PVSGRelationAnnotation(f'{data_dir}/pvsg.json', split)
# collect video id within the split
video_list = pvsg_dataset.video_ids
start_time = time.time()

for vid in video_list:
    # if not os.path.exists(f'{work_dir}/{vid}/relations.pickle'):
    if True:
        print(f'start processing: {vid} at {time.time() - start_time:.2f}s',
              flush=True)
        query_feats = load_pickle(f'{work_dir}/{vid}/query_feats.pickle')

        # obtain matched_tubes
        pred_mask_tubes = get_pred_mask_tubes_one_video(vid, work_dir)
        # gt_mask_tubes = get_gt_mask_tubes_one_video(vid, pvsg_dataset, data_dir)
        # matching_dict = match_tubes(gt_mask_tubes, pred_mask_tubes)
        matching_dict = match_and_process_gt_tubes(vid, pvsg_dataset,
                                                   pred_mask_tubes)
        matching_dict = compact_matching_dict(matching_dict)

        # assign relations
        gt_relations = pvsg_dataset[vid]['relations']
        pred_relations = translate_gt_relations(matching_dict, gt_relations)

        # matching queries
        pred_feat_tubes = {}
        for idx, query_feat in enumerate(query_feats):
            pred_feat_tubes[
                query_feats[idx].track_id] = query_feats[idx].qf_tube

        relation_dict = process_feats_and_relations(pred_relations,
                                                    pred_feat_tubes)
        save_pickle(f'{work_dir}/{vid}/relations.pickle', relation_dict)
