import torch


def pick_top_pairs_eval(pred_matrix, num_total_pairs=100):
    with torch.no_grad():
        # Mask the diagonal elements
        num_objects = pred_matrix.size(0)
        pred_matrix = pred_matrix.clone()
        pred_matrix[torch.eye(num_objects).bool()] = float('-inf')

        pred_matrix_flat = pred_matrix.view(-1)
        max_pairs = min(pred_matrix_flat.size(0), num_total_pairs)

        # Get top 100 predicted pairs
        _, top_indices = torch.topk(pred_matrix_flat, max_pairs, sorted=True)
        num_objects = pred_matrix.size(0)
        top_pairs = [(index // num_objects, index % num_objects)
                     for index in top_indices
                     if index // num_objects != index % num_objects]

        # Remove duplicates and truncate to ensure exactly num_total_pairs
        return [[int(s.item()), int(o.item())] for s, o in top_pairs]


def generate_results(span_pred, prob, selected_pairs):
    results = []

    # Flatten prob to 1D and find the top K values and their indices
    _, sorted_pair_indices = torch.sort(prob.flatten(), descending=True)

    num_relations = prob.size(1)
    for idx in sorted_pair_indices:
        # Map the 1D index back to 2D indices
        pair_index = idx // num_relations
        relation_index = idx % num_relations
        # Get the corresponding pair
        subject_index, object_index = selected_pairs[pair_index]
        # Extract the span prediction for this pair and relation
        relation_span = span_pred[pair_index, :, relation_index].cpu().numpy()

        # Convert the span prediction to binary (0 or 1)
        relation_span_binary = (relation_span > 0).astype(float)

        # Create a result dictionary
        result = {
            'subject_index': subject_index,
            'object_index': object_index,
            'relation': relation_index.item(),
            'relation_span': relation_span_binary
        }
        results.append(result)

    return results


def generate_pairwise_results(span_pred, prob, selected_pairs):
    results = []

    # Find the max probability relation for each pair
    max_probs, max_indices = torch.max(prob, dim=1)

    # Get the top K pairs based on max probabilities
    _, sorted_pair_indices = torch.sort(max_probs, descending=True)

    for pair_idx in sorted_pair_indices:
        relation_index = max_indices[pair_idx]
        subject_index, object_index = selected_pairs[pair_idx]

        # Extract the span prediction for this pair and relation
        relation_span = span_pred[pair_idx, :, relation_index].cpu().numpy()

        # Convert the span prediction to binary (0 or 1)
        relation_span_binary = (relation_span > 0).astype(float)

        # Create a result dictionary
        result = {
            'subject_index': subject_index,
            'object_index': object_index,
            'relation': relation_index.item(),
            'relation_span': relation_span_binary
        }
        results.append(result)

    return results
