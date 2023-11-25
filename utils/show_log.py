import os, csv
import numpy as np


def save_metrics_to_csv(final_metrics, pair_recall_list, K_values,
                        csv_file_path, model_name):
    file_exists = os.path.isfile(csv_file_path)
    # Define the header
    header = ['Model', 'Pair Recall']
    for K in K_values:
        header.extend([f'R/mR@{K}'])
    for K in K_values:
        header.extend([f'wR/wmR@{K}'])

    # Prepare the data row
    data = [model_name, f'{100 * np.array(pair_recall_list).mean():.2f}']
    for K in K_values:
        data.extend([
            f"{100 * final_metrics[K]['recall']:.2f}/{100 * final_metrics[K]['mean_recall']:.2f}",
        ])
    for K in K_values:
        data.extend([
            f"{100 * final_metrics[K]['weak_recall']:.2f}/{100 * final_metrics[K]['weak_mean_recall']:.2f}"
        ])

    # Write the data to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)
