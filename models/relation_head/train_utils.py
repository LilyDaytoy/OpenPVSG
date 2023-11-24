import random
import torch
import torch.nn.functional as F


def rew_bce_loss(y_true, y_pred, class_counts):
    total_counts = class_counts.sum()
    class_weights = total_counts / class_counts
    weighted_loss = F.binary_cross_entropy_with_logits(
        y_pred, y_true, pos_weight=class_weights)
    return weighted_loss


def zlpr_loss(y_true, y_pred):
    """https://kexue.fm/archives/7359."""
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 9999
    y_pred_pos = y_pred - (1 - y_true) * 9999
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    total_loss = neg_loss + pos_loss

    return total_loss.mean()


def pick_top_pairs(gt_relations, pred_matrix, num_total_pairs=100):
    with torch.no_grad():
        pred_matrix_flat = pred_matrix.view(-1)
        max_pairs = min(pred_matrix_flat.size(0), num_total_pairs)

        # Get ground truth pairs
        gt_pairs = [(relation['subject_index'], relation['object_index'])
                    for relation in gt_relations]

        # Get top 100 predicted pairs
        top_scores, top_indices = torch.topk(pred_matrix_flat,
                                             max_pairs - len(gt_pairs),
                                             sorted=True)
        num_objects = pred_matrix.size(0)
        top_pairs = [(index // num_objects, index % num_objects)
                     for index in top_indices
                     if index // num_objects != index % num_objects]

        # Combine top predicted pairs with ground truth pairs
        combined_pairs = gt_pairs + top_pairs

        # Remove duplicates and truncate to ensure exactly num_total_pairs
        selected_pairs = list(set(combined_pairs))

        return [[int(s.item()), int(o.item())] for s, o in selected_pairs]


def get_gt_pairs(gt_relations, num_total_pairs=100):
    gt_pairs = list(
        set([(relation['subject_index'], relation['object_index'])
             for relation in gt_relations]))
    if len(gt_pairs) > num_total_pairs:
        gt_pairs = random.sample(gt_pairs, num_total_pairs)

    return [[int(s.item()), int(o.item())] for s, o in gt_pairs]


def concatenate_sub_obj(sub_feats, obj_feats, selected_pairs):
    concatenated_feats = []

    for subject_idx, object_idx in selected_pairs:
        subject_feat = sub_feats[subject_idx]
        object_feat = obj_feats[object_idx]

        # Concatenate the features along the feature dimension
        concatenated_pair_feat = torch.cat([subject_feat, object_feat], dim=-1)
        concatenated_feats.append(concatenated_pair_feat)

    # Stack all the concatenated features to create the final output tensor
    concatenated_feats = torch.stack(concatenated_feats)

    return concatenated_feats


def generate_gt_matrix(gt_relations, selected_pairs, span_mat_shape,
                       custom_span):

    # Initialize ground truth tensors with zeros
    num_pairs, num_frames, num_relations = span_mat_shape
    gt_span = torch.zeros(num_pairs, num_frames, num_relations)
    gt_prob = torch.zeros(num_pairs, num_relations)

    # Fill in the ground truth tensors
    for relation in gt_relations:
        subject_index = relation['subject_index'].item()
        object_index = relation['object_index'].item()
        relation_index = relation['relation'].item()
        relation_span = relation['relation_span'].squeeze()

        if [subject_index, object_index] in selected_pairs:
            # Find the index of the pair in selected_pairs
            pair_index = selected_pairs.index([subject_index, object_index])

            # Set the ground truth for this pair and relation
            gt_span[
                pair_index, :,
                relation_index] = relation_span[custom_span[0]:custom_span[1]]
            gt_prob[pair_index, relation_index] = 1

    return gt_span, gt_prob


def reshape_and_filter(gt_span, span_pred):
    # Reshape from [batch_size, sequence_length, num_relations] to [batch_size*num_relations, sequence_length]
    gt_span_permuted = gt_span.permute(0, 2, 1)
    span_pred_permuted = span_pred.permute(0, 2, 1)
    gt_span_reshaped = gt_span_permuted.reshape(-1, gt_span_permuted.size(2))
    span_pred_reshaped = span_pred_permuted.reshape(-1,
                                                    span_pred_permuted.size(2))

    # Filter out all-zero vectors in gt_span
    non_zero_indices = torch.any(gt_span_reshaped != 0, axis=1)
    gt_span_filtered = gt_span_reshaped[non_zero_indices]
    span_pred_filtered = span_pred_reshaped[non_zero_indices]

    return gt_span_filtered, span_pred_filtered
