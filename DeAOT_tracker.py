import gc 
import glob 
import os 
import sys 
import time 
import cv2 
import numpy as np 
import torch
from segment_anything import SamPredictor, sam_model_registry

import vot

prj_path = '/usr/mvl2/esdft/aot-sam-main/'
sys.path.append(prj_path)


from helpers import draw_mask, save_prediction
from model_args import aot_args, sam_args, segtracker_args
from SegTracker import SegTracker
from vot_data_preprocessing import get_bbox, get_mask



class DeAOTTracker():

    def __init__(self, segtracker_args, sam_args, aot_args):

        self.seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.seg_tracker.restart_tracker()


    def initialize(self, first_img, objects):

        initial_mask, self.tracks = get_initial_masks(objects, first_img)

        with torch.cuda.amp.autocast():
            initial_mask, tracks = get_initial_masks(objects, first_img)
            self.seg_tracker.add_reference(first_img, initial_mask)
            torch.cuda.empty_cache()
            gc.collect()


    def track(self, frame):

        with torch.cuda.amp.autocast():

            
            pred_mask = self.seg_tracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            pred_list = []
            
            for obj_idx in self.tracks:
                obj_mask = (pred_mask == obj_idx).astype(np.uint8)
                pred_list.append(obj_mask)

        return pred_list





def get_initial_masks(objects, first_img):

    initial_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
    tracks = {}

    for i, gt_mask in enumerate(objects, 1):

        x1, y1, w, h = get_bbox(gt_mask)
        initial_mask[y1:y1+h, x1:x1+w] = gt_mask[y1:y1+h, x1:x1+w] * i
        tracks[i] = []

    return initial_mask, tracks







