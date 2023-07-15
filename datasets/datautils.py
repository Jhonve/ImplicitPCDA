import os

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from utils.utils import get_camera_parameters, label_processing, scale_to_unit_cube, write_pc
from libs.fps.fps_utils import farthest_point_sampling

k_classes_pointdan = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']

k_objs_graspnet = [0, 2, 5, 8, 9, 15, 46, 48, 60, 66]
k_scenes_list_graspnet_train = [[0, 1, 2, 39],
                                [3, 4, 5, 18, 19],
                                [0, 1, 2, 18, 19],
                                [6, 7, 8, 15],
                                [6, 7, 8, 30],
                                [0, 1, 2, 15, 16],
                                [0, 1, 21, 25],
                                [0, 1, 2, 42],
                                [3, 4, 5, 33],
                                [0, 1, 27, 28]]
k_scenes_list_graspnet_test = [[122],
                               [107],
                               [107],
                               [108],
                               [100],
                               [99],
                               [87],
                               [101],
                               [103],
                               [129]]

k_camera_param_kinect = get_camera_parameters(camera='kinect')
k_intrinsic_mat_kinect = k_camera_param_kinect.intrinsic.intrinsic_matrix

k_camera_param_realsense = get_camera_parameters(camera='realsense')
k_intrinsic_mat_realsense = k_camera_param_realsense.intrinsic.intrinsic_matrix

def saveH5(files_path, h5_path, target_name='datapath.h5'):
    files_path = np.array(files_path)

    with h5py.File(h5_path + target_name, 'w') as data_file:
        data_type = h5py.special_dtype(vlen=str)
        data = data_file.create_dataset('datapath', files_path.shape, dtype=data_type)
        data[:] = files_path
        data_file.close()

    print('Save path done!')

def convert_2_point_cloud(self, depth_img, depth_scale, intrinsic_mat):
    pixel_height, pixel_width = depth_img.shape

    dense_points = np.zeros((pixel_height, pixel_width, 3))
    assert(depth_img.shape[0] == pixel_height and depth_img.shape[1] == pixel_width)

    dense_points[:, :, 2] = depth_img / depth_scale
    dense_points[:, :, 0] = np.arange(1, pixel_width + 1)
    temp_arange = np.zeros((pixel_width, pixel_height, 1))
    temp_arange[:, :, 0] = np.arange(1, pixel_height + 1)
    dense_points[:, :, 1] = temp_arange.T

    dense_points[:, :, 0] = (dense_points[:, :, 0] - intrinsic_mat[0][2]) * dense_points[:, :, 2] / intrinsic_mat[0][0]
    dense_points[:, :, 1] = (dense_points[:, :, 1] - intrinsic_mat[1][2]) * dense_points[:, :, 2] / intrinsic_mat[1][1]

    dense_points = np.reshape(dense_points, (pixel_height * pixel_width, 3))
    zero_indices = np.where(dense_points == [0., 0., 0.])
    zero_indices = zero_indices[0]
    zero_indices = np.reshape(zero_indices, (-1, 3))
    zero_indices = zero_indices[:, 0]

    dense_points = np.delete(dense_points, zero_indices, 0)
    
    if dense_points.shape[0] < 1024:
        return np.array([None]), np.array([None])

    # dense_points = pc_preprocessing(dense_points)
    dense_points = scale_to_unit_cube(dense_points)

    dense_points = farthest_point_sampling(dense_points, self.opt.points_num)

    return dense_points

def filtered_GraspNet_real(data_path, write_path, fake_data_path=None, mode='Kinect', phase='train', depth_scale=1000):
    objs_list = k_objs_graspnet
    if phase == 'train':
        scenes_list = k_scenes_list_graspnet_train
    else:
        scenes_list = k_scenes_list_graspnet_test
    all_files_count = 0

    for i_obj in range(len(objs_list)):
        for j_scene in range(len(scenes_list[i_obj])):
            folder_name = 'scene_' + str(scenes_list[i_obj][j_scene]).zfill(4) + '/' + mode + '/'
            file_list_depth = os.listdir(data_path + folder_name + 'depth/')
            for k_file in range(len(file_list_depth)):
                file_rgb = data_path + folder_name + 'rgb/' + file_list_depth[k_file]
                file_depth = data_path + folder_name + 'depth/' + file_list_depth[k_file]
                file_label = data_path + folder_name + 'label/' + file_list_depth[k_file]

                frame_idx = file_list_depth[k_file].split('.')[0]

                if fake_data_path == None:
                    background_mask = label_processing(file_label,  objs_list[i_obj])
                    foreground_mask = 1 - background_mask.astype(np.int)
                    frgd_sum = np.sum(np.sum(foreground_mask, axis=1), axis=0)
                    if frgd_sum < 1024:
                        continue
                else:
                    paired_fake_file_label = fake_data_path + folder_name + 'synthetic_res/' + frame_idx + '_label.png'
                    background_mask = label_processing(file_label,  objs_list[i_obj])
                    paired_background_mask = label_processing(paired_fake_file_label,  objs_list[i_obj])
                    background_mask = np.logical_or(background_mask, paired_background_mask).astype(np.int)
                    foreground_mask = 1 - background_mask
                    frgd_sum = np.sum(np.sum(foreground_mask, axis=1), axis=0)
                    if frgd_sum < 1024:
                        continue
                
                depth_img = cv2.imread(file_depth, 2)
                depth_img[background_mask] = 0

                if mode == 'Kinect':
                    intrinsic_mat = k_intrinsic_mat_kinect
                else:
                    intrinsic_mat = k_intrinsic_mat_realsense
                dense_points = convert_2_point_cloud(depth_img, depth_scale, intrinsic_mat)
                if dense_points.all() == None:
                    continue
                all_files_count += 1

                # write point clouds
                scene_name = folder_name.split('/')[0]
                frame_name = file_list_depth[k_file].split('.')[0]
                output_pc_path = os.path.join(write_path, mode, str(i_obj), scene_name + '_' + frame_name + '.xyz')
                write_pc(dense_points, output_pc_path)

    print('For real-scan data, camera %s, %s stage, get %d point clouds'%(mode, phase, all_files_count))

def filtered_GraspNet_fake(data_path, write_path, real_data_path, mode='/kinect', phase='train', depth_scale=1000):
    objs_list = k_objs_graspnet
    if phase == 'train':
        scenes_list = k_scenes_list_graspnet_train
    else:
        scenes_list = k_scenes_list_graspnet_test
    all_files_count = 0

    for i_obj in range(len(objs_list)):
        for j_scene in range(len(scenes_list[i_obj])):
            folder_name = 'scene_' + str(scenes_list[i_obj][j_scene]).zfill(4) + mode + '/synthetic_res/'
            file_list_all = os.listdir(data_path + folder_name)
            for k_file in range(len(file_list_all)):
                if not 'depth' in file_list_all[k_file]:
                    continue
                
                file_depth = data_path + folder_name + file_list_all[k_file]
                frame_idx = file_list_all[k_file].split('_')[0]
                file_rgb = data_path + folder_name + frame_idx + '_rgb.png'
                file_label = data_path + folder_name + frame_idx + '_label.png'

                paired_real_file_label = real_data_path + 'scene_' + str(scenes_list[i_obj][j_scene]).zfill(4) + mode + '/'\
                                         + 'label/' + frame_idx + '.png'

                background_mask = label_processing(file_label,  objs_list[i_obj])
                paired_background_mask = label_processing(paired_real_file_label,  objs_list[i_obj])
                background_mask = np.logical_or(background_mask, paired_background_mask).astype(np.int)
                foreground_mask = 1 - background_mask
                frgd_sum = np.sum(np.sum(foreground_mask, axis=1), axis=0)
                if frgd_sum < 1024:
                    continue

                depth_img = cv2.imread(file_depth, 2)
                depth_img[background_mask] = 0

                if mode == 'Kinect':
                    intrinsic_mat = k_intrinsic_mat_kinect
                else:
                    intrinsic_mat = k_intrinsic_mat_realsense
                dense_points = convert_2_point_cloud(depth_img, depth_scale, intrinsic_mat)
                if dense_points.all() == None:
                    continue
                all_files_count += 1

                # write point clouds
                scene_name = folder_name.split('/')[0]
                frame_name = frame_idx
                output_pc_path = os.path.join(write_path, mode, str(i_obj), scene_name + '_' + frame_name + '.xyz')
                write_pc(dense_points, output_pc_path)

    print('For synthetic data, camera %s, %s stage, get %d point clouds'%(mode, phase, all_files_count))

def datapath_PointDAN_synthetic(datapath, folder_list, is_train):
    all_file_list = []
    for i in range(len(folder_list)):
        pc_list = os.listdir(datapath + folder_list[i] + '/train/')
        for j in range(len(pc_list)):
            file_npy = datapath + folder_list[i] + '/train/' + pc_list[j]
            file_label = folder_list[i]
            file_int_label = k_classes_pointdan.index(file_label)
            all_file_list.append(file_npy + '|' + str(file_int_label))
    
    return all_file_list

def datapath_prepare(opt):
    if opt.dataset == 'PointDAN':
        folder_list_fake = os.listdir(opt.datapath_fake)
        print('Number of fake classes: ', len(folder_list_fake))

        files_path_fake = datapath_PointDAN_synthetic(opt.datapath_fake, folder_list_fake, opt.is_train)
        print('Number of fake data: ', len(files_path_fake))
        saveH5(files_path_fake, opt.datapath_h5, opt.datapath_file_fake.split('/')[-1])