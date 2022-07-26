import os
import glob

import h5py
import numpy as np
import open3d as o3d

import torch.utils.data as data

from utils.utils import jitter_pointcloud, scale_to_unit_cube, rotate_pc, random_rotate_one_axis, jitter_pointcloud_adaptive
from libs.fps.fps_utils import farthest_point_sampling

class PointDANDataset(data.Dataset):
    def __init__(self, opt, datapath_fake, datapath_real):
        super(PointDANDataset).__init__()
        self.opt = opt
        self.is_train = self.opt.is_train
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_threads
        self.datapath_fake = datapath_fake
        
        self.real_pcs = []
        self.real_labels = []
        data_real_list = ['train_0.h5', 'train_1.h5', 'train_2.h5']
        for f in data_real_list: 
            h5_file_path = os.path.join(datapath_real, f)
            h5_file = h5py.File(h5_file_path, 'r')
            f_data = h5_file['data'][:]
            f_label = h5_file['label'][:]
            h5_file.close()

            f_data = np.squeeze(f_data)
            for i, f_d in enumerate(f_data):
                self.real_pcs.append(f_d.astype(np.float32))
                self.real_labels.append(f_label[i])

        self.fake_len = len(self.datapath_fake)
        self.real_len = len(self.real_pcs)

        if self.fake_len < self.real_len:
            self.SIZE = self.real_len
        else:
            self.SIZE = self.fake_len

    def __len__(self):
        return self.SIZE

    def load_pc_npy(self, datapath):
        if type(datapath) == bytes:
            datapath = datapath.decode('UTF-8')
        datapath_npy = datapath.split('|')[0]
        label = int(datapath.split('|')[1])

        pc = np.load(datapath_npy)
        return pc, label
    
    def __getitem__(self, index):
        if self.fake_len < self.real_len:
            real_index = index
            fake_index = index % self.fake_len
        else:
            fake_index = index
            real_index = index % self.real_len
        
        data_item = {}
        fake_pc, fake_label = self.load_pc_npy(self.datapath_fake[fake_index])
        fake_pc = scale_to_unit_cube(fake_pc)

        if(self.opt.rotation_augmentation):
            random_rotate_one_axis(fake_pc, 'z')
        fake_pc = jitter_pointcloud_adaptive(fake_pc)

        data_item['FPC'] = fake_pc
        fake_pc = farthest_point_sampling(fake_pc, self.opt.points_num)
        data_item['FPCfps'] = fake_pc
        data_item['FLabel'] = fake_label

        real_pc = scale_to_unit_cube(self.real_pcs[real_index][:, 0:3])
        real_pc = rotate_pc(real_pc)
        
        if(self.opt.rotation_augmentation):
            random_rotate_one_axis(real_pc, 'z')
        real_pc = jitter_pointcloud_adaptive(real_pc)

        data_item['RPC'] = real_pc # if modelnet
        real_pc = farthest_point_sampling(real_pc, self.opt.points_num)
        # data_item['RPC'] = real_pc # if shapenet, do fps firstly
        data_item['RPCfps'] = real_pc
        data_item['RLabel'] = self.real_labels[real_index]
        
        data_item['FPaths'] = self.datapath_fake[fake_index]
        data_item['RPaths'] = real_index

        return data_item

    def get_data_loader(self):
        return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

class PointDANBothSyntheticDataset(data.Dataset):
    def __init__(self, opt, datapath_model, datapath_shape):
        super(PointDANBothSyntheticDataset).__init__()
        self.opt = opt
        self.is_train = self.opt.is_train
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_threads
        self.datapath_model = datapath_model
        self.datapath_shape = datapath_shape

        self.model_len = len(self.datapath_model)
        self.shape_len = len(self.datapath_shape)

        if self.model_len < self.shape_len:
            self.SIZE = self.shape_len
        else:
            self.SIZE = self.model_len
    def __len__(self):
        return self.SIZE

    def load_pc_npy(self, datapath):
        if type(datapath) == bytes:
            datapath = datapath.decode('UTF-8')
        datapath_npy = datapath.split('|')[0]
        label = int(datapath.split('|')[1])

        pc = np.load(datapath_npy)
        return pc, label
    
    def __getitem__(self, index):
        if self.model_len < self.shape_len:
            shape_index = index
            model_index = index % self.model_len
        else:
            model_index = index
            shape_index = index % self.shape_len
        
        data_item = {}
        model_pc, model_label = self.load_pc_npy(self.datapath_model[model_index])
        model_pc = scale_to_unit_cube(model_pc)

        if(self.opt.rotation_augmentation):
            random_rotate_one_axis(model_pc, 'z')
        model_pc = jitter_pointcloud_adaptive(model_pc)   # jitter when test

        data_item['FPC'] = model_pc
        model_pc = farthest_point_sampling(model_pc, self.opt.points_num)
        data_item['FPCfps'] = model_pc
        data_item['FLabel'] = model_label

        shape_pc, shape_label = self.load_pc_npy(self.datapath_model[model_index])
        shape_pc = scale_to_unit_cube(shape_pc)
        shape_pc = rotate_pc(shape_pc) # rotate for scannet and modelnet
        
        if(self.opt.rotation_augmentation):
            random_rotate_one_axis(shape_pc, 'z')
        shape_pc = jitter_pointcloud_adaptive(shape_pc)   # jitter when test

        shape_pc = farthest_point_sampling(shape_pc, self.opt.points_num)
        data_item['RPC'] = shape_pc
        data_item['RPCfps'] = shape_pc
        data_item['RLabel'] = shape_label
        
        data_item['FPaths'] = self.datapath_model[model_index]
        data_item['RPaths'] = self.datapath_shape[shape_index]

        return data_item

    def get_data_loader(self):
        return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
    
class GraspNetPointClouds(data.Dataset):
    def __init__(self, dataroot, partition='train'):
        super(GraspNetPointClouds).__init__()
    def __getitem__(self, item):
        o3d_pointcloud = o3d.io.read_point_cloud(self.pc_list[item])
        pointcloud = np.asarray(o3d_pointcloud.points)

        pointcloud = pointcloud.astype(np.float32)
        path = self.pc_list[item].split('.x')[0]
        label = np.copy(self.label[item])

        data_item = {}
        data_item['PC'] = pointcloud
        data_item['Label'] = label
        data_item['Paths'] = path

        return data_item

    def __len__(self):
        return len(self.pc_list)

    def get_data_loader(self, batch_size, num_workers, drop_last, shuffle=True):
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)

class GraspNetRealPointClouds(GraspNetPointClouds):
    def __init__(self, dataroot, mode, partition='train'):
        super(GraspNetRealPointClouds).__init__()
        self.partition = partition

        DATA_DIR = os.path.join(dataroot, partition, "Real", mode) # mode can be 'kinect' or 'realsense'
        # read data
        xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))

        self.pc_list = []
        self.lbl_list = []

        for xyz_path in xyzs_list:
            self.pc_list.append(xyz_path)
            self.lbl_list.append(int(xyz_path.split('/')[-2]))

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

class GraspNetSynthetictPointClouds(GraspNetPointClouds):
    def __init__(self, dataroot, partition='train', device=None):
        super(GraspNetSynthetictPointClouds).__init__()
        self.partition = partition

        if device == None:
            DATA_DIR_kinect = os.path.join(dataroot, partition, "Synthetic", "kinect")
            DATA_DIR_realsense = os.path.join(dataroot, partition, "Synthetic", "realsense")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR_kinect, '*', '*.xyz')))
            xyzs_list_realsense = sorted(glob.glob(os.path.join(DATA_DIR_realsense, '*', '*.xyz')))

            xyzs_list.extend(xyzs_list_realsense)
        elif device == 'kinect':
            DATA_DIR = os.path.join(dataroot, partition, "Synthetic", "kinect")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))
        elif device == 'realsense':
            DATA_DIR = os.path.join(dataroot, partition, "Synthetic", "realsense")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))

        self.pc_list = []
        self.lbl_list = []

        for xyz_path in xyzs_list:
            self.pc_list.append(xyz_path)
            self.lbl_list.append(int(xyz_path.split('/')[-2]))

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)