import sys, argparse
from utils.relation_matching import *
sys.path.append('/mnt/lustre/jkyang/CVPR23/openpvsg')
from tqdm import tqdm
import multiprocessing

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

pvsg_dataset = PVSGRelationAnnotation(f'{data_dir}/pvsg.json', split)
video_list = pvsg_dataset.video_ids


def process_video(vid):
    try:
        query_feats = load_pickle(f'{work_dir}/{vid}/query_feats.pickle')
        pred_mask_tubes = get_pred_mask_tubes_one_video(vid, work_dir)
        matching_dict = match_and_process_gt_tubes(vid, pvsg_dataset,
                                                   pred_mask_tubes)
        matching_dict = compact_matching_dict(matching_dict)
        gt_relations = pvsg_dataset[vid]['relations']
        pred_relations = translate_gt_relations(matching_dict, gt_relations)
        pred_feat_tubes = {
            query_feats[idx].track_id: query_feats[idx].qf_tube
            for idx in range(len(query_feats))
        }
        relation_dict = process_feats_and_relations(pred_relations,
                                                    pred_feat_tubes)
        save_pickle(f'{work_dir}/{vid}/relations.pickle', relation_dict)
        print(f'{vid} Completed!', flush=True)
    except Exception as e:
        print(f'An error occurred with video {vid}: {e}')
    return vid


with multiprocessing.Pool(processes=10) as pool:
    # You can replace tqdm with pool.imap or pool.imap_unordered if you want a progress bar
    results = list(
        tqdm(pool.imap(process_video, video_list), total=len(video_list)))
