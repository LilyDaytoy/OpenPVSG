import os, glob, json
from pathlib import Path

import copy
from turtle import pd

import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from datasets.datasets.utils import PVSGAnnotation, vpq_eval


@DATASETS.register_module()
class PVSGImageDataset:
    def __init__(self,
                 pipeline=None,
                 data_root='./data/',
                 annotation_file='pvsg.json',
                 test_mode=False,
                 split='train',
                 with_relation: bool = False):
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

        # collect video id within the split
        video_ids, img_names = [], []
        for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
            for video_id in anno['split'][data_source][split]:
                video_ids.append(video_id)
                img_names += glob.glob(
                    os.path.join(data_root, data_source, 'frames', video_id,
                                 '*.png'))

        assert anno_file.exists()
        assert data_root.exists()

        # get annotation file
        anno = PVSGAnnotation(anno_file, video_ids)

        # find all images
        images = []
        for itm in img_names:
            vid = itm.split(sep='/')[-2]
            vid_anno = anno[vid]

            images.append({
                'img': itm,
                'ann': itm.replace('frames', 'masks'),
                'objects': vid_anno['objects'],
                'pre_hook': self.cates2id,
            })

            assert os.path.exists(
                images[-1]
                ['img']), f"File not found: {split, images[-1]['img']}"
            assert os.path.exists(
                images[-1]
                ['ann']), f"File not found: {split, images[-1]['ann']}"

        self.images = images
        # self.images = images[:16] # for debug

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

    def cates2id(self, category):
        class2ids = dict(
            zip(self.CLASSES + self.BACKGROUND_CLASSES,
                range(len(self.CLASSES + self.BACKGROUND_CLASSES))))
        return class2ids[category]

    def pre_pipelines(self, results):
        results['img_info'] = []
        results['thing_lower'] = 0
        results['thing_upper'] = self.num_thing_classes
        results['ori_filename'] = os.path.basename(results['img'])
        results['filename'] = results['img']

    def prepare_train_img(self, idx):
        results = copy.deepcopy(self.images[idx])
        self.pre_pipelines(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.images[idx])
        self.pre_pipelines(results)
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
            return self.prepare_train_img(idx)

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
