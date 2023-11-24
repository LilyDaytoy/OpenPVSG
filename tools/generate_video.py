# srun -p regular python tools/generate_result_video.py

import cv2
import os

# Directory containing the images
work_dir = 'train_save_qf'
video_id = '0001_4164158586'

image_folder = f'./work_dirs/{work_dir}/{video_id}/qualititive'
video_name = f'./work_dirs/{work_dir}/{video_id}/{video_id}.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort(key=lambda x: int(x.split('.')[0])
            )  # Sort the images by filename numerically if needed

# Assuming that all images are the same size, get the dimensions of the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 5.0, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
