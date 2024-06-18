import os
import cv2
import torch
import numpy as np

def plot_image(img, label_img=None, save_file='image.png'):
    # if img is tensor, convert to cv2 image
    if torch.is_tensor(img):
        img = img.mul(255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    if label_img is not None:
        # if label_img is tensor, convert to cv2 image
        if torch.is_tensor(label_img):
            color_label_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            color_label_img[...,2] = label_img.mul(255.0).cpu().numpy().astype(np.uint8)
            label_img = color_label_img
        # overlay images
        img = cv2.addWeighted(label_img, 0.3, img, 1.0, 0)
    # save image
    cv2.imwrite(save_file, img)

def add_module_prefix(state_dict):
    """Adds the 'module.' prefix to the keys of the state dict."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict['module.' + k] = v
    return new_state_dict

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg