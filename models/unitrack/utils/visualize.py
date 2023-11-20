
import cv2
import numpy as np
import imageio as io
from matplotlib import cm

import time
import PIL

import pycocotools.mask as mask_utils
from . import palette

CLASSES = ['adult', 'baby', 'bag', 'ball', 'ballon', 'basket', 'bat', 'bed', 
'bench', 'beverage', 'bike', 'bird', 'blanket', 'board', 'book', 'bottle', 
'bowl', 'box', 'bread', 'brush', 'bucket', 'cabinet', 'cake', 'camera', 'can', 
'candle', 'car', 'card', 'carpet', 'cart', 'cat', 'cellphone', 'chair', 
'child', 'chopstick', 'cloth', 'computer', 'condiment', 'cookie', 'countertop', 
'cover', 'cup', 'curtain', 'dog', 'door', 'drawer', 'dustbin', 'egg', 'fan', 
'faucet', 'fence', 'flower', 'fork', 'fridge', 'fruit', 'gift', 'glass', 
'glasses', 'glove', 'grain', 'guitar', 'hat', 'helmet', 'horse', 'iron', 
'knife', 'light', 'lighter', 'mat', 'meat', 'microphone', 'microwave', 'mop', 
'net', 'noodle', 'others', 'oven', 'pan', 'paper', 'piano', 'pillow', 'pizza', 
'plant', 'plate', 'pot', 'powder', 'rack', 'racket', 'rag', 'ring', 'scissor', 
'shelf', 'shoe', 'simmering', 'sink', 'slide', 'sofa', 'spatula', 'sponge', 
'spoon', 'spray', 'stairs', 'stand', 'stove', 'switch', 'table', 'teapot', 
'towel', 'toy', 'tray', 'tv', 'vaccum', 'vegetable', 'washer', 'window', 
'ceiling', 'floor', 'grass', 'ground', 'rock', 'sand', 'sky', 'snow', 
'tree', 'wall', 'water']


def dump_predictions(pred, lbl_set, img, prefix):
    '''
    Save:
        1. Predicted labels for evaluation
        2. Label heatmaps for visualization
    '''
    lbl_set = palette.tensor.astype(np.uint8)
    sz = img.shape[:-1]

    # Upsample predicted soft label maps
    # pred_dist = pred.copy()
    pred_dist = cv2.resize(pred, sz[::-1])[:]
    
    # Argmax to get the hard label for index
    pred_lbl = np.argmax(pred_dist, axis=-1)
    pred_lbl = np.array(lbl_set, dtype=np.int32)[pred_lbl]      
    mask = np.float32(pred_lbl.sum(2) > 0)[:,:,None]
    alpha = 0.5
    img_with_label = mask * (np.float32(img) * alpha + \
            np.float32(pred_lbl) * (1-alpha)) + (1-mask) * np.float32(img)

    # Visualize label distribution for object 1 (debugging/analysis)
    pred_soft = pred_dist[..., 1]
    pred_soft = cv2.resize(pred_soft, (img.shape[1], img.shape[0]), 
            interpolation=cv2.INTER_NEAREST)
    pred_soft = cm.jet(pred_soft)[..., :3] * 255.0
    img_with_heatmap1 =  np.float32(img) * 0.5 + np.float32(pred_soft) * 0.5

    # Save blend image for visualization
    io.imwrite('%s_blend.jpg' % prefix, np.uint8(img_with_label))

    if prefix[-4] != '.':  # Super HACK-y
        imname2 = prefix + '_mask.png'
    else:
        imname2 = prefix.replace('jpg','png')

    # Save predicted labels for evaluation
    io.imwrite(imname2, np.uint8(pred_lbl))

    return img_with_label, pred_lbl, img_with_heatmap1

def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        video = (video*255).astype(np.uint8)
        
    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration = 0.2)

def get_color(idx):
    idx = idx * 17
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, obs, obj_ids, scores=None, frame_id=0, fps=0.):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 150.))
    alpha = 0.4

    for i, ob in enumerate(obs): 
        obj_id = int(obj_ids[i])
        id_text = '{}_{}'.format(CLASSES[ob['class_id']], str(obj_id))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(obj_id)
        if len(ob) == 4:
            x1, y1, w, h = ob
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
        elif isinstance(ob, dict):
            mask = mask_utils.decode(ob)
            mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)[:,:,None]
            mask_color = mask * color
            im = (1 - mask) * im + mask * (alpha*im + (1-alpha)*mask_color)
            # put the id tag at the mask centroid
            M = cv2.moments(mask)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = mask.shape[1]//2, mask.shape[0]//2
            (text_width, text_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)
            tag = np.zeros((text_height + 10, text_width + 10, 3), dtype=np.uint8)
            cv2.putText(tag, id_text, (5, text_height + 5), cv2.FONT_HERSHEY_PLAIN, text_scale, (255,255,255), thickness=text_thickness)
            # put the tag on image
            start_x = cx - tag.shape[1] // 2
            start_y = cy - tag.shape[0] // 2
            tag_height, tag_width = tag.shape[:2]
            start_x = max(start_x, 0)
            start_y = max(start_y, 0)
            end_x = start_x + tag_width
            end_y = start_y + tag_height
            if end_x > im.shape[1]:
                end_x = im.shape[1]
                tag_width = im.shape[1] - start_x
            if end_y > im.shape[0]:
                end_y = im.shape[0]
                tag_height = im.shape[0] - start_y
            im[start_y:end_y, start_x:end_x] = tag[:tag_height, :tag_width]

        else:
            raise ValueError('Observation format not supported.')
    return im


def vis_pose(oriImg, points):

    pa = np.zeros(15)
    pa[2] = 0
    pa[12] = 8
    pa[8] = 4
    pa[4] = 0
    pa[11] = 7
    pa[7] = 3
    pa[3] = 0
    pa[0] = 1
    pa[14] = 10
    pa[10] = 6
    pa[6] = 1
    pa[13] = 9
    pa[9] = 5
    pa[5] = 1

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[0, :]
    y = points[1, :]

    for n in range(len(x)):
        pair_id = int(pa[n])

        x1 = int(x[pair_id])
        y1 = int(y[pair_id])
        x2 = int(x[n])
        y2 = int(y[n])

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            cv2.line(canvas, (x1, y1), (x2, y2), colors[n], 8)

    return canvas


def draw_skeleton(aa, kp, color, show_skeleton_labels=False, dataset= "PoseTrack"):
    if dataset == "COCO":
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], 
                [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',
                    'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    elif dataset == "PoseTrack":
        skeleton = [[10, 11], [11, 12], [9,8], [8,7],
                    [10, 13], [9, 13], [13, 15], [10,4],
                    [4,5], [5,6], [9,3], [3,2], [2,1]]
        kp_names = ['right_ankle', 'right_knee', 'right_pelvis',
                    'left_pelvis', 'left_knee', 'left_ankle',
                    'right_wrist', 'right_elbow', 'right_shoulder',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'upper_neck', 'nose', 'head']
    for i, j in skeleton:
        if kp[i-1][0] >= 0 and kp[i-1][1] >= 0 and kp[j-1][0] >= 0 and kp[j-1][1] >= 0 and \
            (len(kp[i-1]) <= 2 or (len(kp[i-1]) > 2 and  kp[i-1][2] > 0.1 and kp[j-1][2] > 0.1)):
            st = (int(kp[i-1][0]), int(kp[i-1][1]))
            ed = (int(kp[j-1][0]), int(kp[j-1][1]))
            cv2.line(aa, st, ed,  color, max(1, int(aa.shape[1]/150.)))
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:
            pt = (int(kp[j][0]), int(kp[j][1]))
            if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                cv2.circle(aa, pt, 2, tuple((0,0,255)), 2)
            elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                cv2.circle(aa, pt, 2, tuple((255,0,0)), 2)

            if show_skeleton_labels and (len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1)):
                cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
