import os
import time
import datetime

import torch
import numpy as np

from datasets.datautils import datapath_prepare
from datasets.datasets import PointDANDataset, GraspNetRealPointClouds, GraspNetSynthetictPointClouds

from options.train_options import TrainOptions

from models import create_model

from tensorboardX import SummaryWriter

def train_graspnet(opt, train_dataloader_A, train_dataloader_B, num_train_batch, model):
    tsboard_writer = SummaryWriter('runs/' + opt.name)

    total_iters = 0
    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration

        i_train = 0
        for data_A, data_B in zip(train_dataloader_A, train_dataloader_B):
            iter_start_time = time.time()   # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            if total_iters % opt.print_freq == 0:
                optimize_start_time = time.time()

            if epoch == opt.epoch_count and i_train == 0:
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()

            data = {}
            data['FPCfps'] = data_A['PC']
            data['FLabel'] = data_A['Label']
            data['FPaths'] = data_A['Paths']

            data['RPCfps'] = data_B['PC']
            data['RLabel'] = data_B['Label']
            data['RPaths'] = data_B['Paths']


            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            if total_iters % opt.print_freq == 0:
                optimize_time = time.time() - optimize_start_time

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                for loss_label, loss_value in losses.items():
                    tsboard_writer.add_scalar(loss_label, loss_value, global_step=(epoch - 1) * num_train_batch + i_train + 1)
                
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            if total_iters % opt.print_freq == 0:
                iter_data_time = time.time()

                eta = (optimize_time + t_data) * num_train_batch * (opt.n_epochs + opt.n_epochs_decay - epoch) +\
                    (optimize_time + t_data) * (num_train_batch - i_train - 1)
                eta = str(datetime.timedelta(seconds=int(eta)))

                print("Epoch: %d/%d; Batch: %d/%d, ETA: %s (%.4fs opt. %.4fs load)" % 
                        (epoch, opt.n_epochs + opt.n_epochs_decay, i_train + 1, num_train_batch, 
                            eta, optimize_time, t_data))

            i_train += 1

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

if __name__ == '__main__':
    opt = TrainOptions().parse()

    if opt.datapath_prepared == False:
        datapath_prepare(opt)

    if not os.path.exists(opt.checkpoints_dir + '/' + opt.name):
        os.makedirs(opt.checkpoints_dir + '/' + opt.name)

    train_dataset_A = GraspNetSynthetictPointClouds('../Dataset/GraspNetPointClouds/', 'train')
    train_dataset_B = GraspNetRealPointClouds('../Dataset/GraspNetPointClouds/', opt.camera_mode, 'train')
    train_dataloader_A = train_dataset_A.get_data_loader(opt.batch_size, opt.num_threads, drop_last=True, shuffle=True)
    train_dataloader_B = train_dataset_B.get_data_loader(opt.batch_size, opt.num_threads, drop_last=True, shuffle=True)
    
    # initialize Network structure etc.
    network_model = create_model(opt)

    num_train_batch = int(min(train_dataset_A.__len__(), train_dataset_B.__len__()) / opt.batch_size)

    # start train
    train_graspnet(opt, train_dataloader_A, train_dataloader_B, num_train_batch, network_model)