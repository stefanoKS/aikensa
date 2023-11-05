import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import cv2
import torch

from dataclasses import dataclass
from YOLOv6.yolov6.core.inferer import Inferer
from YOLOv6.yolov6.utils.nms import non_max_suppression

@dataclass
class EngineConfig:
    webcam:bool = False
    webcam_addr:str ='0'
    img_size:int =1280
    weights:str = './weights/yolov6s6.pt'
    device:int = 0
    yaml:str = './YOLOv6/data/coco.yaml'
    conf_thres:float = 0.4
    iou_thres:float = 0.45
    max_det:int = 1000

def create_inferer(config:EngineConfig):
    inferer = Inferer(None, config.webcam, config.webcam_addr, config.weights, config.device, config.yaml, config.img_size, False)
    return inferer

@torch.inference_mode()
def custom_infer_stream(inferer, conf_thres=0.4, iou_thres=0.45, max_det=1000):
    for img_src, img_path, vid_cap in tqdm(inferer.files):
        img, img_src = inferer.process_image(img_src, inferer.img_size, inferer.stride, inferer.half)
        img = img.to(inferer.device)
        if len(img.shape) == 3:
            img = img[None]
        pred_results = inferer.model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, None, False, max_det=max_det)[0]

        if len(det):
            det[:, :4] = inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)  # integer class
                label = f'{inferer.class_names[class_num]} {conf:.2f}'
                inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=inferer.generate_colors(class_num, True))

            img_src = np.asarray(img_src)

@torch.inference_mode()
def custom_infer_single(inferer, img_arr, conf_thres=0.4, iou_thres=0.45, max_det=1000):
    img, img_src = inferer.process_image(img_arr, inferer.img_size, inferer.stride, inferer.half)
    img = img.to(inferer.device)
    if len(img.shape) == 3:
        img = img[None]
    pred_results = inferer.model(img)
    det = non_max_suppression(pred_results, conf_thres, iou_thres, None, False, max_det=max_det)[0]

    if len(det):
        det[:, :4] = inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
        for *xyxy, conf, cls in reversed(det):
            class_num = int(cls)  # integer class
            label = f'{inferer.class_names[class_num]} {conf:.2f}'
            inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=inferer.generate_colors(class_num, True))

        img_src = np.asarray(img_src)
    
    return det, img_src