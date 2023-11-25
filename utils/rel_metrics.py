import os
import numpy as np
from PIL import Image


def calculate_iou(span1, span2):
    intersection = (span1 * span2).sum()
    union = span1.sum() + span2.sum() - intersection
    return intersection / union if union > 0 else 0


def calculate_mask_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0
    else:
        return intersection / union


def calculate_pair_recall_at_k(selected_pairs, gt_pairs, k=20):
    selected_pairs_set = set(tuple(pair) for pair in selected_pairs[:k])
    gt_pairs_set = set(tuple(pair) for pair in gt_pairs)
    correct_predictions = len(selected_pairs_set.intersection(gt_pairs_set))
    recall = correct_predictions / len(gt_pairs_set) if gt_pairs_set else 0
    return recall


def calculate_final_metrics(relation_recall_dict, K_values):
    final_metrics = {}
    num_valid_rel = len([
        rel for rel in relation_recall_dict[K_values[0]].values()
        if rel['total'] != 0
    ])
    for K in K_values:
        total_recall = sum(rel['hit']
                           for rel in relation_recall_dict[K].values())
        total_weak_recall = sum(rel['weak_hit']
                                for rel in relation_recall_dict[K].values())
        total_gt = sum(rel['total']
                       for rel in relation_recall_dict[K].values())
        recall_at_k = total_recall / total_gt if total_gt > 0 else 0
        weak_recall_at_k = total_weak_recall / total_gt if total_gt > 0 else 0
        mean_recall = sum(rel['hit'] / rel['total']
                          for rel in relation_recall_dict[K].values()
                          if rel['total'] != 0) / num_valid_rel
        weak_mean_recall = sum(rel['weak_hit'] / rel['total']
                               for rel in relation_recall_dict[K].values()
                               if rel['total'] != 0) / num_valid_rel
        final_metrics[K] = {
            'recall': recall_at_k,
            'mean_recall': mean_recall,
            'weak_recall': weak_recall_at_k,
            'weak_mean_recall': weak_mean_recall,
        }
    return final_metrics


def calculate_viou(gt_set, pred_set, vid, data_dir):
    (gt_sub_idx, gt_obj_idx, gt_span_list) = gt_set
    (pred_sub_mask_list, pred_obj_mask_list, pred_span_list) = pred_set
    pred_sub_mask_dict, pred_obj_mask_dict = {}, {}
    for mask_dict in pred_sub_mask_list:
        for k, v in mask_dict.items():
            pred_sub_mask_dict[k] = v
    for mask_dict in pred_obj_mask_list:
        for k, v in mask_dict.items():
            pred_obj_mask_dict[k] = v

    # Determine the data source
    if vid.startswith('P'):
        data_source = 'epic_kitchen'
    elif vid.split('_')[0].isdigit() and len(vid.split('_')[0]) == 4:
        data_source = 'vidor'
    else:
        data_source = 'ego4d'

    masks_root = os.path.join(data_dir, data_source, 'masks', vid)

    gt_real_span_list = np.zeros_like(pred_span_list)
    pred_hit_list = np.zeros_like(pred_span_list)

    for range_pair in gt_span_list:
        start, end = range_pair
        for frame_id in range(start, end + 1):
            if frame_id >= len(pred_span_list):
                continue
            mask_path = os.path.join(masks_root,
                                     str(frame_id).zfill(4) + '.png')
            pan_mask = np.array(Image.open(mask_path))
            gt_sub_mask = (pan_mask == gt_sub_idx).astype(bool)
            gt_obj_mask = (pan_mask == gt_obj_idx).astype(bool)
            # clean up the gt list
            if np.any(gt_sub_mask) and np.any(gt_obj_mask):
                gt_real_span_list[frame_id] = 1
            if frame_id in pred_sub_mask_dict and frame_id in pred_obj_mask_dict:
                # process subjects
                pred_sub_mask = pred_sub_mask_dict[frame_id].numpy().astype(
                    bool)
                sub_iou = calculate_mask_iou(gt_sub_mask, pred_sub_mask)
                # process objects
                pred_obj_mask = pred_obj_mask_dict[frame_id].numpy().astype(
                    bool)
                obj_iou = calculate_mask_iou(gt_obj_mask, pred_obj_mask)
                if sub_iou >= 0.5 and obj_iou >= 0.5:
                    pred_hit_list[frame_id] = 1
    # compute overlapping_ones
    pred_hit_list_real = np.logical_and(
        pred_hit_list == 1, pred_span_list == 1).astype(pred_hit_list.dtype)

    # iou is the standard protocol, iou_weak ignores time span evaluation
    iou_weak = calculate_iou(pred_hit_list, gt_real_span_list)
    iou = calculate_iou(pred_hit_list_real, gt_real_span_list)

    return iou, iou_weak
