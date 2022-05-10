import os

import cv2
import h5py
import numpy as np
import open3d as o3d

import torch.utils.data as data

from utils.utils import get_camera_parameters, jitter_pointcloud, lable_cropping, label_processing, pc_preprocessing, scale_to_unit_cube, rotate_pc, random_rotate_one_axis, jitter_pointcloud_adaptive
from libs.fps.fps_utils import farthest_point_sampling

k_classes_graspnet_all = [1, 3, 6, 8, 9, 10, 19, 21, 23, 27, 30, 35, 37, 38, 39, 41, 42, 44, 45, 47, 49, 52, 53, 57, 58, 59, 61, 62, 63, 67, 70, 71]
k_classes_pointdan = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']

