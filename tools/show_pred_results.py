import torch, os, cv2
import numpy as np
from torch.utils.data import DataLoader
from datasets import PVSGRelationDataset
from models.relation_head.base import ObjectEncoder, PairProposalNetwork
from models.relation_head.base import VanillaModel
from models.relation_head.convolution import HandcraftedFilter, Learnable1DConv
from models.relation_head.transformer import TemporalTransformer
from models.relation_head.train_utils import concatenate_sub_obj
from models.relation_head.test_utils import pick_top_pairs_eval, \
    generate_results, generate_pairwise_results
from utils.rel_metrics import calculate_pair_recall_at_k, calculate_final_metrics, calculate_viou
from utils.show_log import save_metrics_to_csv
from utils.relation_matching import PVSGRelationAnnotation, get_pred_mask_tubes_one_video
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the dataset and dataloader
data_dir = './data/'
split = 'val'
ps_type = 'ips'
model_name = 'transformer'  # vanilla, filter, conv, transformer
model_pth = 'epoch_100.pth'
mark = 'full_result'

mark = model_name + '_' + mark + '_' + model_pth.split('.')[0]
work_dir = f'./work_dirs/{ps_type}_{split}_save_qf'
save_work_dir = f'./work_dirs/relation/rel_{ps_type}_{model_name}'
loaded_state_dicts = torch.load(os.path.join(save_work_dir, model_pth))

pvsg_rel_dataset = PVSGRelationDataset(f'{data_dir}/pvsg.json',
                                       split,
                                       work_dir,
                                       return_mask=True)
data_loader = DataLoader(pvsg_rel_dataset, batch_size=1, shuffle=False)
object_names = pvsg_rel_dataset.classes
relation_list = pvsg_rel_dataset.relations

pvsg_ann_dataset = PVSGRelationAnnotation(f'{data_dir}/pvsg.json', split)

# for pairing
feature_dim = 256
hidden_dim = 1024

# for relation network
input_dim = 512

# for dataset
num_relations = 57
num_top_pairs = 100
max_frame_length = 900
accumulation_steps = 8

# load models
subject_encoder = ObjectEncoder(feature_dim=feature_dim)
subject_encoder.load_state_dict(loaded_state_dicts['subject_encoder'])
subject_encoder.to(device).eval()

object_encoder = ObjectEncoder(feature_dim=feature_dim)
object_encoder.load_state_dict(loaded_state_dicts['object_encoder'])
object_encoder.to(device).eval()

pair_proposal_model = PairProposalNetwork(feature_dim, hidden_dim)
pair_proposal_model.load_state_dict(loaded_state_dicts['pair_proposal_model'])
pair_proposal_model.to(device).eval()

assert model_name in ['vanilla', 'filter', 'conv', 'transformer'], \
    f'Model {model_name} unsupported'

model_classes = {
    'vanilla': VanillaModel,
    'filter': HandcraftedFilter,
    'conv': Learnable1DConv,
    'transformer': TemporalTransformer
}
if model_name in model_classes:
    relation_model = model_classes[model_name](input_dim,
                                               num_relations).to(device)
else:
    raise ValueError(f'Model {model_name} is unsupported')

relation_model.load_state_dict(loaded_state_dicts['relation_model'])
relation_model.to(device).eval()


def show_video(subject_encoder, object_encoder, pair_proposal_model,
               relation_model, data_loader, num_top_pairs, relation_list,
               device, work_dir, save_work_dir, mark):

    subject_encoder.eval()
    object_encoder.eval()
    pair_proposal_model.eval()
    relation_model.eval()

    for i, relation_dict in enumerate(data_loader):
        with torch.no_grad():
            vid = relation_dict['vid'][0]
            feats = relation_dict['feats'][0].float().to(device)
            num_frames = len(feats[0])

            # Convert the features into subjects or objects
            sub_feats = subject_encoder(feats)
            obj_feats = object_encoder(feats)

            # Forward pass through the Pair Proposal Network
            pred_matrix = pair_proposal_model(sub_feats, obj_feats)

            # get top pairs
            selected_pairs = pick_top_pairs_eval(pred_matrix, num_top_pairs)

            # evaluate the performance of pair selecting
            concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats,
                                                     selected_pairs)
            span_pred, prob = relation_model(concatenated_feats)

            # results = generate_results(span_pred, prob, selected_pairs)
            results = generate_pairwise_results(span_pred, prob,
                                                selected_pairs)

            idx2key = relation_dict['idx2key']
            for key in idx2key:
                idx2key[key] = idx2key[key].item()

            object_dict = {}
            for idx, mask_dict in enumerate(relation_dict['masks']):
                if len(mask_dict) > 0:
                    object_dict[idx] = object_names[int(mask_dict['cid'][0])]

            triplet_set = []
            for idx, result in enumerate(results[:20]):
                s_id, o_id = result['subject_index'], result['object_index']
                if s_id in object_dict and o_id in object_dict:
                    s_name, o_name = object_dict[s_id], object_dict[o_id]
                    s_id_map, o_id_map = idx2key[s_id], idx2key[o_id]
                    relation_name = relation_list[result['relation']]
                    triplet_str = f'{s_name}-{s_id_map} {relation_name} {o_name}-{o_id_map}'
                    print(triplet_str, flush=True)
                    triplet_set.append({
                        'triplet': triplet_str,
                        'span': result['relation_span'],
                    })
            torch.cuda.empty_cache()
            del concatenated_feats, span_pred, prob

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_fps = 5
        sample_img = f'{work_dir}/{vid}/qualititive/0000.png'
        image = Image.open(sample_img)
        image_width, image_height = image.size
        chart_height = 15 + len(triplet_set) * 13
        os.makedirs(f'{save_work_dir}/vis/', exist_ok=True)
        output_video_path = f'{save_work_dir}/vis/{vid}.mp4'
        videoWriter = cv2.VideoWriter(
            output_video_path, fourcc, video_fps,
            (int(image_width), int(image_height + chart_height)))

        # draw the progress bar chart
        for frame_idx in range(num_frames):
            sample_img = f'{work_dir}/{vid}/qualititive/{str(frame_idx).zfill(4)}.png'
            image = Image.open(sample_img)
            image_width, image_height = image.size
            chart_height = 15 + len(triplet_set) * 13
            chart = Image.new('RGB', (image_width, chart_height),
                              (255, 255, 255))
            draw_bar = ImageDraw.Draw(chart)
            font = ImageFont.truetype('./assets/OpenSans-Bold.ttf', size=10)

            # Draw the dot and time tags
            bar_height = 5
            draw_bar.rectangle([0, 0, image_width, bar_height], fill=(0, 0, 0))
            dot_interval = image_width / 10
            time_interval = num_frames / 10
            for i in range(11):
                dot_x = i * dot_interval
                dot_y = bar_height / 2
                draw_bar.ellipse([dot_x - 1, dot_y - 1, dot_x + 1, dot_y + 1],
                                 fill=(255, 255, 255))
                tag_x = dot_x - 21
                tag_y = dot_y - 2
                time = time_interval * i
                draw_bar.text((tag_x, tag_y),
                              f'{time:.1f}',
                              fill=(255, 0, 0),
                              font=font)

            # Draw the relation bar
            bar_height = 10
            very_start_time = 0
            for i, triplet_info in enumerate(triplet_set):
                triplet = triplet_info['triplet']
                span = triplet_info['span']
                start_height = 10 + (bar_height + 3) * i
                start_time = None
                for frame_index, frame_value in enumerate(span):
                    if frame_value == 1:
                        if start_time is None:
                            start_time = frame_index / num_frames * image_width
                            very_start_time = frame_index / num_frames * image_width
                        if frame_index == len(span) - 1 or span[frame_index +
                                                                1] == 0:
                            end_time = (frame_index +
                                        1) / num_frames * image_width
                            draw_bar.rectangle([
                                start_time, start_height, end_time,
                                start_height + bar_height
                            ],
                                               fill='#BDE0FE')
                            start_time = None

                text_width = draw_bar.textlength(triplet)
                if very_start_time + text_width <= image_width:
                    draw_bar.text((very_start_time, start_height),
                                  triplet,
                                  fill='#d90429')
                else:
                    draw_bar.text((image_width - text_width, start_height),
                                  triplet,
                                  fill='#d90429')

            # Draw the time position indicator
            indicator_x = frame_idx / num_frames * image_width
            draw_bar.line([indicator_x, 0, indicator_x, chart_height],
                          fill=(255, 0, 0),
                          width=2)

            # concatenating figures
            chart = np.array(chart, dtype=np.uint8)
            image = np.array(image, dtype=np.uint8)
            concat_img = np.concatenate([image, chart], axis=0)
            videoWriter.write(concat_img[..., [2, 1, 0]])
        videoWriter.release()


if __name__ == '__main__':
    show_video(subject_encoder, object_encoder, pair_proposal_model,
               relation_model, data_loader, num_top_pairs, relation_list,
               device, work_dir, save_work_dir, mark)
