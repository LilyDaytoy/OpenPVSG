# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import cv2
import pickle
import numpy as np
import os.path as osp
import pycocotools.mask as mask_utils
from models.unitrack.utils.log import logger
from models.unitrack.utils.meter import Timer
from models.unitrack.utils import visualize as vis
from models.unitrack.utils import io as io

class SimpleTracker(object):
    def __init__(self, track_id, qf_tube):
        self.qf_tube = []
        self.track_id = track_id
        self.qf_tube = qf_tube

def concat_seq(outputs, save_root):
    save_dir = osp.join(save_root, "qualititive")
    io.mkdir_if_missing(save_dir)
    timer = Timer()
    results = []
    object_list = []
    feat_tubes_dict = {}
    for frame_id, output in enumerate(outputs):
        output = output[0]  # {'pan_results': (480,640), 'query_feats': {'1044': [torch.Size(256,)]}}
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1./max(1e-5, timer.average_time)))
        online_ids = []
        online_masks = []
        if len(output['query_feats']) == 0: # nothing in this frame!
            results.append((frame_id + 1, [], [], []))
        else: # run tracking
            timer.tic()
            for ins_id, feat in output['query_feats'].items():
                if ins_id not in object_list:
                    object_list.append(ins_id)
                    feat_tubes_dict[object_list.index(ins_id) + 1] = {}
                tid = object_list.index(ins_id) + 1
                feat_tubes_dict[tid][frame_id] = {
                    'query_feat': feat[0].to(dtype=torch.float32).cpu().numpy(),
                    'cls_id': int(ins_id % 1000)
                    }
                mask = (output['pan_results'] == ins_id).astype(np.uint8)
                mask = mask_utils.encode(np.asfortranarray(mask))
                mask['counts'] = mask['counts'].decode('ascii')
                mask['class_id'] = ins_id % 1000
                online_ids.append(tid)
                online_masks.append(mask)
            timer.toc()
            results.append((frame_id + 1, None, online_masks, online_ids))
        if save_dir is not None:
            vid = save_root.split('/')[-1]
            # Determine the data source
            if vid.startswith('P'):
                data_source = 'epic_kitchen'
            elif vid.split('_')[0].isdigit() and len(vid.split('_')[0]) == 4:
                data_source = 'vidor'
            else:
                data_source = 'ego4d'
                
            gt_image_dir = os.path.join('./data', data_source, 'frames', vid)
            img0 = cv2.imread(os.path.join(gt_image_dir, f'{str(frame_id).zfill(4)}.png'))
            online_im = vis.plot_tracking(img0, online_masks, online_ids, frame_id=frame_id)
            cv2.imwrite(os.path.join(
                save_dir, '{:04d}.png'.format(frame_id)), online_im)

    result_filename = osp.join(save_root, "quantitive/masks.txt")
    io.write_mots_results(result_filename, results)

    # process feat_tubes_dict
    query_feat_tubes = []
    for track_id, feat_tubes in feat_tubes_dict.items():
        qf_tube = []
        for i in range(len(outputs)):
            if i in feat_tubes.keys():
                qf_tube.append(feat_tubes[i])
            else:
                qf_tube.append(None)
        tracker = SimpleTracker(track_id, qf_tube)
        query_feat_tubes.append(tracker)
    # import pdb; pdb.set_trace();
    qf_results_filename = osp.join(save_root, "query_feats.pickle")
    print("Writing results to {}".format(qf_results_filename), flush=True)
    with open(qf_results_filename, "wb") as f:
        pickle.dump(query_feat_tubes, f)


# by HB
# not check
def preprocess_video_panoptic_gt(
        gt_labels,
        gt_masks,
        gt_semantic_seg,
        gt_instance_ids,
        num_things,
        num_stuff,
        img_metas,
):
    num_classes = num_things + num_stuff
    num_frames = len(img_metas)

    thing_masks_list = []
    for frame_id in range(num_frames):
        thing_masks_list.append(gt_masks[frame_id].pad(
            img_metas[frame_id]['pad_shape'][:2], pad_val=0).to_tensor(
            dtype=torch.bool, device=gt_labels.device))
    instances = torch.unique(gt_instance_ids[:, 1])
    things_masks = []
    labels = []
    for instance in instances:
        pos_ins = torch.nonzero(torch.eq(gt_instance_ids[:, 1], instance), as_tuple=True)[0]  # 0 is for redundant tuple
        labels_instance = gt_labels[:, 1][pos_ins]
        assert torch.allclose(labels_instance, labels_instance[0])
        labels.append(labels_instance[0])
        instance_frame_ids = gt_instance_ids[:, 0][pos_ins].to(dtype=torch.int32).tolist()
        instance_masks = []
        for frame_id in range(num_frames):
            frame_instance_ids = gt_instance_ids[gt_instance_ids[:, 0] == frame_id, 1]
            if frame_id not in instance_frame_ids:
                empty_mask = torch.zeros(
                    (img_metas[frame_id]['pad_shape'][:2]),
                    dtype=thing_masks_list[frame_id].dtype, device=thing_masks_list[frame_id].device
                )
                instance_masks.append(empty_mask)
            else:
                pos_inner_frame = torch.nonzero(torch.eq(frame_instance_ids, instance), as_tuple=True)[0].item()
                frame_mask = thing_masks_list[frame_id][pos_inner_frame]
                instance_masks.append(frame_mask)
        things_masks.append(torch.stack(instance_masks))

    things_masks = torch.stack(things_masks)
    things_masks = things_masks.to(dtype=torch.long)
    labels = torch.stack(labels)
    labels = labels.to(dtype=torch.long)

    return labels, things_masks

