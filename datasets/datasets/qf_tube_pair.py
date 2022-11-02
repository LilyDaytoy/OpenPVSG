import os
import copy
from pathlib import Path
from click import utils

from mmdet.datasets.builder import DATASETS
from .utils import PVSGAnnotation
from ...utils.postprocess import load_pred_mask_tubes, load_gt_mask_tubes, \
                                 match_gt_pred_mask_tubes, load_pickle, get_labels_single_video

@DATASETS.register_module()
class QueryFeaturePairDataset:
    def __init__(self,
                 data_root="/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/data/pvsg_demo",
                 split="train",
                 mode="random", # if a whole video in a batch
                 results_root="/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/work_dirs/test_default_ckpt4_tracking_and_save_qf2",
                 anno_file_path="/mnt/lustre/jkyang/wxpeng/CVPR23/pvsg_data/pvsg_demo.json"):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.gt_mask_root = self.data_root / split / "masks"
        self.results_root = Path(results_root) # results from train or test (different process)
        self.gt = PVSGAnnotation(anno_file_path)
        self.vids = os.listdir(self.gt_mask_root)

        self.data = []
        self.labels = []
        if split == "train":
            for vid in self.vids:
                # some special cases in demo, will be deleted -------------------------------------------
                if self.gt[vid]['relations'] is None:
                    continue
                if vid == "1007_6631583821" or vid == "0046_11919433184": # this vid relation is not ready yet
                    continue
                # ---------------------------------------------------------------------------------------
                print("Processing video {}".format(vid), flush=True)
                pairs_filtered_this_video, labels_this_video = self.assign_labels_to_qf_tubes_single_video(vid)
                if len(labels_this_video) == 0:
                    continue
                if mode == "random": # one data is a pair in one frame
                    for tube_pair, tube_label in zip(pairs_filtered_this_video, labels_this_video):
                        for frame_s, frame_o, frame_label in zip(tube_pair[0], tube_pair[1], tube_label['label']):
                            if len(frame_label) == 0:
                                continue
                            self.data.append([frame_s, frame_o])
                            self.labels.append(frame_label)
                else: # one data is a pair tube in one whole video
                    for tube_pair, tube_label in zip(pairs_filtered_this_video, labels_this_video):
                        self.data.append(tube_pair)
                        self.labels.append(tube_label['label'])
        else:
            pass # TODO
                




    
    def assign_labels_to_qf_tubes_single_video(self, vid):
        pred_mask_path = self.results_root / vid / "quantitive/masks.txt"
        pred_mask_tubes = load_pred_mask_tubes(pred_mask_path)
        gt_mask_path = self.gt_mask_root / vid
        gt_mask_tubes = load_gt_mask_tubes(gt_mask_path)
        assigned_labels, iou_scores = match_gt_pred_mask_tubes(pred_mask_tubes, gt_mask_tubes)
        qf_tube_obj_list = load_pickle(self.results_root / vid / "query_feats.pickle")
        gt_relations = self.gt[vid]['relations']
        pairs_filtered, labels = get_labels_single_video(assigned_labels, qf_tube_obj_list, gt_relations)
        return pairs_filtered, labels

        
    def prepare_train_img(self, idx):
        return {'data': self.data[idx],
                'label': self.labels[idx]}

    def __getitem__(self, idx):
        if self.split == "train":
            return self.prepare_train_img(idx)


    def __len__(self):
        return len(self.data)