# srun -p regular python frame_extraction.py

import os
from tqdm import tqdm

video_root = '../data/ego4d/videos'
save_root = '../data/ego4d/frames'


def make_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


video_list = [
    'c20407ac-83d6-4c84-88cb-63bced9d456b.mp4',
    'e127fc34-0de5-41b0-ab68-7d5574bcf613.mp4',
    '22cc4d54-34be-4580-983a-9e710e831c16.mp4',
]

# for video_name in tqdm(os.listdir(video_root)):
for video_name in tqdm(video_list):
    video_dir = os.path.join(video_root, video_name)
    vid = video_name.split('.')[0]
    print('Extracting frames for video {}'.format(vid))
    save_dir = os.path.join(save_root, vid)
    make_dir_if_not_exist(save_dir)
    os.system('ffmpeg -i {} -start_number 0 {}/%04d.png'.format(
        video_dir, save_dir))
