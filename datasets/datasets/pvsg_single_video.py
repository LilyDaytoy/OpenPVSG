import os
from pathlib import Path

import json
import copy

import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from datasets.datasets.utils import SeqObj, PVSGAnnotation, vpq_eval


@DATASETS.register_module()
class PVSGSingleVideoImageDataset:
    """A dataset only used for test to connect tracker to get query feature
    tube of a video."""
    def __init__(
        self,
        pipeline=None,
        data_root='./data/',
        annotation_file='pvsg.json',
        video_name='0010_8610561401',
        test_mode=False,
        split='test',
    ):
        assert data_root is not None
        data_root = Path(data_root)
        anno_file = data_root / annotation_file

        with open(anno_file, 'r') as f:
            anno = json.load(f)

        # collect class names
        self.THING_CLASSES = anno['objects']['thing']  # 115
        self.STUFF_CLASSES = anno['objects']['stuff']  # 11
        self.BACKGROUND_CLASSES = ['background']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.num_thing_classes = len(self.THING_CLASSES)
        self.num_stuff_classes = len(self.STUFF_CLASSES)
        self.num_classes = len(self.CLASSES)  # 126

        if video_name.startswith('P'):
            data_source = 'epic_kitchen'
        elif video_name.split('_')[0].isdigit() and len(
                video_name.split('_')[0]) == 4:
            data_source = 'vidor'
        else:
            data_source = 'ego4d'

        # eg. ./data/pvsg_demo/val/images/0010_8610561401
        images_dir = data_root / data_source / 'frames' / video_name
        img_names = sorted([str(x) for x in (images_dir.rglob('*.png'))])

        # find all images
        images = []
        for frame_id, itm in enumerate(img_names):
            images.append({
                'video_name': video_name,
                'frame_id': frame_id,
                'img': itm,
            })

            assert os.path.exists(images[-1]['img'])

        self.images = images  # "data" of this dataset
        # self.images = images[:16] # for debug
        # self.images = images[104:250] # debug

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

    def pre_pipelines(self, results):
        results['img_info'] = []
        results['thing_lower'] = 0
        results['thing_upper'] = self.num_thing_classes
        results['ori_filename'] = os.path.basename(results['img'])
        results['filename'] = results['img']

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = copy.deepcopy(self.images[idx])  # eg. [0,0,1]
        self.pre_pipelines(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.images[idx])
        self.pre_pipelines(results)
        # if not self.ref_seq_index:
        #     results = results[0]
        return self.pipeline(results)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    # Copy and Modify from mmdet
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            while True:
                cur_data = self.prepare_train_img(idx)
                if cur_data is None:
                    idx = self._rand_another(idx)
                    continue
                return cur_data

    def __len__(self):
        return len(self.images)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)

    def evaluate(self, results,
                 **kwargs):  # can only support evaluation in order now !
        max_ins = 10000  # same as self.divisor
        pq_results = []
        pipeline = Compose([dict(type='LoadAnnotationsDirect')])
        for idx, _result in enumerate(results):
            img_info = self.images[idx]
            self.pre_pipelines(img_info)
            gt = pipeline(img_info)
            gt_pan = gt['gt_panoptic_seg'].astype(np.int64)
            pan_seg_result = copy.deepcopy(_result['pan_results'])
            pan_seg_map = -1 * np.ones_like(pan_seg_result)
            for itm in np.unique(pan_seg_result):
                if itm >= INSTANCE_OFFSET:
                    cls = itm % INSTANCE_OFFSET
                    ins = itm // INSTANCE_OFFSET
                    pan_seg_map[pan_seg_result == itm] = cls * max_ins + ins
                elif itm == self.num_classes:
                    pan_seg_map[pan_seg_result ==
                                itm] = self.num_classes * max_ins
                else:
                    pan_seg_map[pan_seg_result == itm] = itm * max_ins
            assert -1 not in pan_seg_result
            pq_result = vpq_eval([pan_seg_map, gt_pan],
                                 num_classes=self.num_classes,
                                 max_ins=max_ins,
                                 ign_id=self.num_classes)
            pq_results.append(pq_result)

        iou_per_class = np.stack([result[0] for result in pq_results
                                  ]).sum(axis=0)[:self.num_classes]
        tp_per_class = np.stack([result[1] for result in pq_results
                                 ]).sum(axis=0)[:self.num_classes]
        fn_per_class = np.stack([result[2] for result in pq_results
                                 ]).sum(axis=0)[:self.num_classes]
        fp_per_class = np.stack([result[3] for result in pq_results
                                 ]).sum(axis=0)[:self.num_classes]
        epsilon = 0.
        sq = iou_per_class / (tp_per_class + epsilon)
        rq = tp_per_class / (tp_per_class + 0.5 * fn_per_class +
                             0.5 * fp_per_class + epsilon)
        pq = sq * rq
        pq = np.nan_to_num(pq)  # set nan to 0.0
        return {
            'PQ': pq,
            'PQ_all': pq.mean(),
            'PQ_th': pq[:self.num_thing_classes].mean(),
            'PQ_st': pq[self.num_thing_classes:].mean(),
        }
