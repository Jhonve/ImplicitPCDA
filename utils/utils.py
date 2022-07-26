import cv2
import numpy as np
import open3d as o3d

import torch

def label_processing(label_path, label_id):
    label = cv2.imread(label_path)
    label_max = np.max(label)

    if label_max == 255: # is rendered synthetic label
        label = label[:, :, 0]
        foreground_mask = (label == label_id + 1)
        background_mask = 1 - foreground_mask
    else:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        foreground_mask = (label == label_id + 1)
        background_mask = 1 - foreground_mask

    return background_mask.astype(np.bool8)

def label_background_processing(label_path):
    label = cv2.imread(label_path)
    label_max = np.max(label)

    if label_max == 255: # is rendered synthetic label
        label = label[:, :, 0]
        background_mask = (label >= 90) # there are 88 models in total, reduce rasterization error
    else:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        background_mask = (label <= 0)

    return background_mask

def lable_cropping(label_path, label_id, bias=8):
    label_img = cv2.imread(label_path)
    h = label_img.shape[0]
    w = label_img.shape[1]

    idx_h, idx_w, _ = np.where(label_img == label_id + 1)

    if idx_h.shape[0] == 0: # no such label
        up = int((h / 2) - 128)
        down = int((h / 2) + 128)
        left = int((w / 2) - 128)
        right = int((w / 2) + 128)
        return up, down, left, right

    up = np.min(idx_h) - bias
    down = np.max(idx_h) + bias
    left = np.min(idx_w) - bias
    right = np.max(idx_w) + bias

    if right - left >= down - up:
        mid_h = (up + down) / 2
        len = right - left
        up = int(mid_h - (len / 2))
        down = int(mid_h + (len / 2))

        # check bbox
        if up < 0:
            up = 0
            down = up + len
        elif down >= h:
            down = h - 1
            up = down - len

        if left < 0:
            left = 0
            right = left + len
        elif right >= w:
            right = w - 1
            left = right - len
    else:
        mid_w = (left + right) / 2
        len = down - up
        left = int(mid_w - (len / 2))
        right = int(mid_w + (len / 2))

        # check bbox
        if up < 0:
            up = 0
            down = up + len
        elif down >= h:
            down = h - 1
            up = down - len

        if left < 0:
            left = 0
            right = left + len
        elif right >= w:
            right = w - 1
            left = right - len

    return up, down, left, right

def label_processing_cropping(label_path, label_id, bias=8):
    label_img = cv2.imread(label_path)
    h = label_img.shape[0]
    w = label_img.shape[1]

    bg_h, bh_w, _ = np.where(label_img == 0)
    if bg_h.shape[0] >= 16:
        # is real mask
        up, down, left, right = lable_cropping(label_path, label_id=254)
    else:
        up, down, left, right = lable_cropping(label_path, label_id * 3 - 1)

    return up, down, left, right

def get_camera_parameters(extrinsic_mat=None, camera=''):
    param = o3d.camera.PinholeCameraParameters()
    
    if extrinsic_mat == None:
        param.extrinsic = np.eye(4, dtype=np.float64)
    else:
        param.extrinsic = extrinsic_mat

    # param.intrinsic = o3d.camera.PinholeCameraIntrinsic()

    if 'kinect' in camera:
        param.intrinsic.set_intrinsics(1280, 720, 631.5, 631.2, 639.5, 359.5)
    elif 'realsense' in camera:
        param.intrinsic.set_intrinsics(1280, 720, 927.17, 927.37, 639.5, 359.5)
    else:
        print("Unknow camera type")
        exit(0)

    return param

def scale_to_unit_cube(x):
    if len(x) == 0:
        return x
    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x

def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')

def rotate_pc(pc):
    pc = rotate_shape(pc, 'x', -np.pi / 2)
    return pc

def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')

def translate_pointcloud(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud.astype('float32')

def jitter_pointcloud_adaptive(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape

    inner = np.matmul(pointcloud, np.transpose(pointcloud, (1, 0)))
    pc_2 = np.sum(pointcloud ** 2, axis = 1, keepdims=True)
    pairwise_distances = pc_2 - 2 * inner + np.transpose(pc_2, (1, 0))
    zero_mask = np.where(pairwise_distances <= 1e-4)
    pairwise_distances[zero_mask] = 9999.
    min_distances = np.min(pairwise_distances, axis=1)
    min_distances = np.sqrt(min_distances)

    min_distances_expdim = np.expand_dims(min_distances, axis=1)
    min_distances_expdim = np.repeat(min_distances_expdim, C, axis=1)

    # pointcloud += np.clip(min_distances_expdim * np.random.randn(N, C), -1 * min_distances_expdim, min_distances_expdim) # normal sampling
    pointcloud += np.clip(min_distances_expdim * (np.random.rand(N, C) * 2. - 1.), -1 * min_distances_expdim, min_distances_expdim) # uniform sampling
    return pointcloud.astype('float32')

def pc_preprocessing(pc):
    mean = np.mean(pc, axis=0)
    pc = pc - mean
    pc_max = np.max(np.abs(pc))

    pc = pc / (pc_max * 1.1)
    pc = (pc + 1.) / 2.
    return pc

def nearest_distances(x, y):
    # x query, y target
    inner = -2 * torch.matmul(x.transpose(2, 1), y) # x B 3 N; y B 3 M
    xx = torch.sum(x**2, dim=1, keepdim=True)
    yy = torch.sum(y**2, dim=1, keepdim=True)
    
    pairwise_distance = xx.transpose(2, 1) + inner + yy
    nearest_distance = torch.sqrt(torch.min(pairwise_distance, dim=2, keepdim=True).values)

    return nearest_distance

def self_nearest_distances(x):
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # x B 3 N
    xx = torch.sum(x**2, dim=1, keepdim=True)
    
    pairwise_distance = xx.transpose(2, 1) + inner + xx
    pairwise_distance += torch.eye(x.shape[2]).to(pairwise_distance.device) * 2
    nearest_distance = torch.sqrt(torch.min(pairwise_distance, dim=2, keepdim=True).values)

    return nearest_distance

def self_nearest_distances_K(x, k=3):
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # x B 3 N
    xx = torch.sum(x**2, dim=1, keepdim=True)
    
    pairwise_distance = xx.transpose(2, 1) + inner + xx
    pairwise_distance += torch.eye(x.shape[2]).to(pairwise_distance.device) * 2
    pairwise_distance *= -1
    k_nearest_distance = pairwise_distance.topk(k=k, dim=2)[0]
    k_nearest_distance *= -1

    nearest_distance = torch.sqrt(torch.mean(k_nearest_distance, dim=2, keepdim=True))

    return nearest_distance

def write_pc(point_cloud, output_path):
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utilit.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(output_path, point_cloud_o3d)