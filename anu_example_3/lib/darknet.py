#!python3
#-*- coding: utf-8 -*-
"""
yolo detection model

@author: RyuManSAng
@date: 20180920
"""
from ctypes import *
import os
import cv2
import numpy as np
import time
import re

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("best_class_idx", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


hasGPU = True
lib = CDLL("lib/libdarknet.so", RTLD_LOCAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions

def decode_detection(detections, bf_size, af_size):
    decoded = []
    for label, confidence, bbox in detections:
        b = bbox
        x = bf_size[1] * (b[0] / af_size[1])
        y = bf_size[0] * (b[1] / af_size[0])
        w = bf_size[1] * (b[2] / af_size[1])
        h = bf_size[0] * (b[3] / af_size[0])
        bbox = (x, y, w, h)
        #confidence = float(round(confidence, 2))#float(round(confidence * 100, 2))
        confidence = confidence

        decoded.append((str(label), confidence, bbox))
    return decoded

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

class DarknetWrapper:
    net = None
    meta = None

    def __init__(self, configPath, weightPath, namesPath, batch_size=1, gpus=0):
        p = 0
        set_gpu(gpus)

        self.net = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, batch_size)
        self.net_width = network_width(self.net)
        self.net_height = network_height(self.net)
        self.darknet_image = make_image(self.net_width, self.net_height, 3)

        self.meta_names = []
        with open(namesPath) as f:
            try:
                label_list = f.read().strip().splitlines()
                self.meta_names = [x.strip() for x in label_list]
            except TypeError:
                pass
        self.meta_classes = len(self.meta_names)

    def detect(self, frame, thresh=.5, hier_thresh=.5, nms=.6):
        bf_size = frame.shape[:2]
        try:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.net_width, self.net_height), interpolation=cv2.INTER_LINEAR)
            #if bf_size != (self.net_width, self.net_height):
            #    frame = cv2.resize(frame, (self.net_width, self.net_height), interpolation=cv2.INTER_LINEAR)
            af_size = frame.shape[:2]
            copy_image_from_bytes(self.darknet_image, frame.tobytes())

            pnum = pointer(c_int(0))
            pre_detections = predict_image(self.net, self.darknet_image)
            re_predictions = [(name, pre_detections[idx]) for idx, name in enumerate(self.meta_names)]

            detections = get_network_boxes(self.net, self.darknet_image.w, self.darknet_image.h, thresh, hier_thresh, None, 0, pnum, 0)

            num = pnum[0]
            if nms:
                do_nms_sort(detections, num, self.meta_classes, nms)
            predictions = remove_negatives(detections, self.meta_names, num)
            predictions = decode_detection(predictions, bf_size, af_size)
            free_detections(detections, num)
            return sorted(predictions, key=lambda x: x[1])
        except Exception as e:
            return [], []