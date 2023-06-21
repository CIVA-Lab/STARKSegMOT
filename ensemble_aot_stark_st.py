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


import vot

from vot_data_preprocessing import get_bbox, get_mask


def aot():


    #prj_path = '/usr/mvl2/esdft/aot-sam-main/'
    sys.path.append(prj_path)

    from helpers import draw_mask, save_prediction
    from model_args import aot_args, sam_args, segtracker_args
    from SegTracker import SegTracker

    imagefile = handle.frame()

    first_img = cv2.imread(imagefile)
    initial_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
    tracks = {}
    for i, obj in enumerate(objects, 1):

        x1, y1, w, h = get_bbox(obj)
        initial_mask[y1:y1+h, x1:x1+w] = obj[y1:y1+h, x1:x1+w] * i
        tracks[i] = []

    seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
    seg_tracker.restart_tracker()

    

    torch.cuda.empty_cache()
    gc.collect()
    frame_idx = 0

    

    with torch.cuda.amp.autocast():
        
        while True:
            frame = cv2.imread(imagefile)
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

            pred_list = []

            for obj_idx in tracks:
                obj_mask = (pred_mask == obj_idx).astype(np.uint8)
                pred_list.append(obj_mask)

            handle.report(pred_list)
            imagefile = handle.frame()

            if not imagefile:
                break

    torch.cuda.empty_cache()
    gc.collect()




def stark_st():

    stark_path = '/usr/mvl2/esdft/'
    sys.path.append(stark_path)


    import stark_st_tracker as stark_tracker

    imagefile = handle.frame()

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    trackers = [stark_tracker.starkTracker() for object in objects]

    for ind in range(len(trackers)):
        trackers[ind].initialize(image, get_bbox(objects[ind]))

    

    while True:

        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        stark_tracker.predictor.set_image(image)

        pred_list = []

        for tracker in trackers:

            box = tracker.tracker.track(image)['target_bbox']
            input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
            masks, _, _ = stark_tracker.predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=False,)
            temp = masks[0]*1
            mask = temp.astype(np.uint8)

            pred_list.append(mask)

        handle.report(pred_list)
        imagefile = handle.frame()

        if not imagefile:
            break






handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()



if len(objects) > 5:
    aot()
else:
    stark_st()
