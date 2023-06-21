import gc
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

prj_path = '/usr/mvl2/esdft/aot-sam-main/'
sys.path.append(prj_path)

from vot.region import Mask
from vot.region import io as vot_io
from vot.region.io import read_trajectory

from helpers import draw_mask, save_prediction
from model_args import aot_args, sam_args, segtracker_args
from SegTracker import SegTracker

from vot_data_preprocessing import get_bbox, get_mask


def main():
    DATASET = '/usr/mvl2/esdft/development_data/sequences/cat-18'
    # DATASET = '/usr/mvl2/itdfh/dev/vot-development/sequences'
    # SEQ = 'cat'
    seq_name = ''
    seq = os.path.join(DATASET, seq_name)

    imgs_paths = sorted(glob.glob(os.path.join(seq, 'color/*.jpg')))
    gt_paths = sorted(glob.glob(os.path.join(seq, 'groundtruth*.txt')))
    print('gt_paths', gt_paths)
    first_img = cv2.imread(imgs_paths[0])
    initial_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
    tracks = {}
    for i, gt_path in enumerate(gt_paths, 1):
        #curr_obj = read_trajectory(gt_path)[0]
        #bounds = curr_obj.bounds()
        #x1, y1, x2, y2 = bounds
        #initial_mask[y1:y2+1, x1:x2+1] = curr_obj.mask * i
        obj = get_mask(gt_path)
        x1, y1, w, h = get_bbox(obj)
        initial_mask[y1:y1+h, x1:x1+w] = obj[y1:y1+h, x1:x1+w] * i
        tracks[i] = []

    seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
    seg_tracker.restart_tracker()

    

    torch.cuda.empty_cache()
    gc.collect()
    frame_idx = 0

   

    with torch.cuda.amp.autocast():
        
        for img_path in imgs_paths:
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                seg_tracker.add_reference(frame, initial_mask)
                torch.cuda.empty_cache()
                gc.collect()
                pred_mask = initial_mask
                frame_idx += 1
                continue
           
            pred_mask = seg_tracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            print(pred_mask)

            for obj_idx in tracks:
                obj_mask = (pred_mask == obj_idx).astype(np.uint8)



    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
