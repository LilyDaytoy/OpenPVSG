import pickle
import numpy as np
import os
import json
import pickle
import copy
from utils.relation_matching import get_pred_mask_tubes_one_video


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class PVSGRelationDataset:
    def __init__(self,
                 anno_file,
                 split='train',
                 work_dir='./work_dirs/train_save_qf_1106',
                 return_mask=False):
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.video_ids = []
        for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
            for video_id in anno['split'][data_source][split]:
                self.video_ids.append(video_id)

        self.work_dir = work_dir
        self.split = split
        self.classes = anno['objects']['thing'] + anno['objects']['stuff']
        self.relations = anno['relations']
        self.return_mask = return_mask

        self.videos = {}
        for video_anno in anno['data']:
            self.videos[video_anno['video_id']] = video_anno

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        relation_dict = load_pickle(
            os.path.join(self.work_dir, vid, 'relations.pickle'))
        relation_dict['vid'] = vid
        # getting all object features
        feat_list = []
        mapping_dict = {}
        for idx, key in enumerate(relation_dict['feats']):
            feat_list.append(relation_dict['feats'][key])
            mapping_dict[key] = idx
        relation_dict['feats'] = np.array(feat_list)

        # getting relation info
        pair_list = []
        for relation in relation_dict['relations']:
            relation['subject_index'] = mapping_dict[relation['subject_index']]
            relation['object_index'] = mapping_dict[relation['object_index']]
            pair_list.append(
                [relation['subject_index'], relation['object_index']])

        # getting pair info
        relation_dict['pairs'] = pair_list

        if self.return_mask:
            rev_mapping_dict = {v: k for k, v in mapping_dict.items()}
            relation_dict['idx2key'] = rev_mapping_dict
            mask_list = []
            pred_mask_tubes = get_pred_mask_tubes_one_video(vid, self.work_dir)
            for idx in range(len(rev_mapping_dict)):
                key = rev_mapping_dict[idx]
                if key in pred_mask_tubes:
                    mask_list.append(pred_mask_tubes[key])
                else:
                    mask_list.append({})
            relation_dict['masks'] = mask_list

        return relation_dict
