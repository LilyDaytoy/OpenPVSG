# srun -p gpu python tools/unzip_and_extract.py

import os, shutil
from tqdm import tqdm
import zipfile
import os
import cv2
import hashlib
from pathlib import Path
from multiprocessing import Pool

zip_dir = './data_zip/'
target_dir = './data/'

# List all the zip files in the directory
zip_files = {
    'Ego4D/ego4d_videos.zip': {
        'target': 'ego4d/videos',
        'md5': '9334c74f5c831c80774862afa9d3f7f0'
    },
    'Ego4D/ego4d_masks.zip': {
        'target': 'ego4d/masks',
        'md5': '218ce689e1e8284e25b50280a5d29612'
    },
    'EpicKitchen/epic_kitchen_videos.zip': {
        'target': 'epic_kitchen/videos',
        'md5': 'b791a71ef24b14721a7b5041190ba4a3'
    },
    'EpicKitchen/epic_kitchen_masks.zip': {
        'target': 'epic_kitchen/masks',
        'md5': '03757120075de23328a11e56660175f4'
    },
    'VidOR/vidor_videos.zip': {
        'target': 'vidor/videos',
        'md5': 'fcc1a6f54ef60aa16fab335a6270c960'
    },
    'VidOR/vidor_masks.zip': {
        'target': 'vidor/masks',
        'md5': '17bfa5ec13235d86273bc9d067776862'
    },
}


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def makedir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def unzip_files(file, destination):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        for member in zip_ref.infolist():
            filename = os.path.basename(member.filename)
            if not filename:
                continue
            file_path = os.path.join(destination, filename)
            with zip_ref.open(member, 'r') as source, open(file_path,
                                                           'wb') as target:
                shutil.copyfileobj(source, target)


def process_video(video_path, save_root):
    video_name = video_path.name.split('.')[0]
    save_dir = os.path.join(save_root, video_name)
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    count = 0
    while success and count < n_frames:
        success, image = cap.read()
        if success:
            cv2.imwrite(os.path.join(save_dir, '{:04d}.png'.format(count)),
                        image)
            count += 1


def extract_videos(video_root, save_root, num_processes=10):
    video_root = Path(video_root)
    videos = list(video_root.rglob('*.mp4'))
    with Pool(num_processes) as pool:
        pool.starmap(process_video, [(video, save_root) for video in videos])


# unzip files
for filename in zip_files:
    zip_path = os.path.join(zip_dir, filename)
    zip_info = zip_files[filename]

    assert calculate_md5(
        zip_path) == zip_info['md5'], f'{zip_path} md5 mismatches!'

    target_path = os.path.join(target_dir, zip_info['target'])
    makedir_if_not_exist(target_path)
    unzip_files(zip_path, target_path)
    print(filename, 'unzipping completed!', flush=True)

    if 'videos' in filename.split('/')[-1]:
        frame_path = target_path.replace('videos', 'frames')
        extract_videos(target_path, frame_path)
        print(frame_path, 'extraction completed!', flush=True)

# copy pvsg.json to data/
shutil.copy('./data_zip/pvsg.json', './data/')
