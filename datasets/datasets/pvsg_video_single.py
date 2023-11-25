import os, glob, json
import random
from pathlib import Path
from typing import List
from typing_extensions import Literal
import pdb

import copy

import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from datasets.datasets.utils import SeqObj, vpq_eval, PVSGAnnotation


@DATASETS.register_module()
class PVSGVideoSingleVideoDataset:
    """A dataset only used for test to get query feature tube from VPS
    model."""
    def __init__(self,
                 pipeline=None,
                 data_root='./data/',
                 annotation_file='pvsg.json',
                 video_name='0010_8610561401',
                 test_mode=False,
                 split='train',
                 ref_sample_mode: Literal['random', 'sequence',
                                          'test'] = 'sequence',
                 ref_seq_index: List[int] = None,
                 ref_seq_len_test: int = 4,
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
        vid2seq_id = {}
        seq_count = 0
        for frame_id, itm in enumerate(img_names):
            tokens = itm.split(sep='/')
            vid, img_id = tokens[-2], tokens[-1].split(sep='.')[0]

            # map vid to seq_id (seq_id starts from 0)
            if vid in vid2seq_id:
                seq_id = vid2seq_id[vid]
            else:
                seq_id = seq_count
                vid2seq_id[vid] = seq_count
                seq_count += 1

            images.append(
                SeqObj({
                    'video_name': video_name,
                    'frame_id': frame_id,
                    'seq_id': seq_id,
                    'img_id': int(img_id),
                    'img': itm,
                }))

            assert os.path.exists(images[-1]['img'])

        self.vid2seq_id = vid2seq_id

        reference_images = {hash(image): image
                            for image in images}  # image -- all SeqObj

        # ref_seq_index is None means no ref img
        self.ref_sample_mode = ref_sample_mode
        if ref_seq_index is None:
            ref_seq_index = []
        self.ref_seq_index = ref_seq_index

        sequences = []
        if self.ref_sample_mode == 'random':
            for img_cur in images:
                is_seq = True
                seq_now = [img_cur.dict]
                if self.ref_seq_index:
                    for index in random.choices(self.ref_seq_index, k=1):
                        query_obj = SeqObj({
                            'seq_id':
                            img_cur.dict['seq_id'],
                            'img_id':
                            img_cur.dict['img_id'] + index
                        })
                        if hash(query_obj) in reference_images:
                            seq_now.append(
                                reference_images[hash(query_obj)].dict)
                        else:
                            is_seq = False
                            break
                if is_seq:
                    sequences.append(seq_now)
        elif self.ref_sample_mode == 'sequence':
            # In the sequence mode, the first frame is the key frame
            # Note that sequence mode may have multiple pointer to one frame
            for img_cur in images:
                is_seq = True
                seq_now = []
                if self.ref_seq_index:  # [0,1]
                    for index in reversed(self.ref_seq_index):
                        query_obj = SeqObj({
                            'seq_id':
                            img_cur.dict['seq_id'],
                            'img_id':
                            img_cur.dict['img_id'] + index
                        })
                        if hash(query_obj) in reference_images:
                            seq_now.append(
                                copy.deepcopy(
                                    reference_images[hash(query_obj)].dict))
                        else:
                            is_seq = False
                            break
                if is_seq:
                    seq_now.append(copy.deepcopy(img_cur.dict))
                    seq_now.reverse()
                    sequences.append(seq_now)
        elif self.ref_sample_mode == 'test':
            if ref_seq_len_test == 0:
                sequences = [[copy.deepcopy(itm.dict)] for itm in images]
            elif ref_seq_len_test == 1:
                sequences = [[
                    copy.deepcopy(itm.dict),
                    copy.deepcopy(itm.dict)
                ] for itm in images]
            else:
                seq_id_pre = -1
                seq_now = []
                for img_cur in images:
                    seq_id_now = img_cur.dict['seq_id']
                    if seq_id_now != seq_id_pre:
                        seq_id_pre = seq_id_now
                        if len(seq_now) > 0:
                            while len(seq_now) < ref_seq_len_test + 1:
                                seq_now.append(copy.deepcopy(seq_now[-1]))
                            sequences.append(seq_now)
                        seq_now = [
                            copy.deepcopy(img_cur.dict),
                            copy.deepcopy(img_cur.dict)
                        ]
                    elif len(seq_now) % (ref_seq_len_test + 1) == 0:
                        sequences.append(seq_now)
                        seq_now = [
                            copy.deepcopy(img_cur.dict),
                            copy.deepcopy(img_cur.dict)
                        ]
                    else:
                        seq_now.append(copy.deepcopy(img_cur.dict))
        else:
            raise ValueError('{} not supported.'.format(self.ref_sample_mode))

        self.sequences = sequences
        self.images = reference_images

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

    def pre_pipelines(self, results):
        for _results in results:
            _results['img_info'] = []
            _results['thing_lower'] = 0
            _results['thing_upper'] = self.num_thing_classes
            _results['ori_filename'] = os.path.basename(_results['img'])
            _results['filename'] = _results['img']

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = copy.deepcopy(self.sequences[idx])  # eg. [0,0,1]
        self.pre_pipelines(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.sequences[idx])
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
        return len(self.sequences)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)

    def evaluate(self, results, **kwargs):
        pass
