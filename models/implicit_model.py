import os
import time
import numpy as np
import open3d as o3d

import torch
from torch.nn import functional as F

from .base_model import BaseModel
from . import networks

from utils.utils import nearest_distances, self_nearest_distances, self_nearest_distances_K

class IMPDECModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for Hole prediction model
        """
        parser.add_argument('--backbone', type=str, default='dgcnn', help='the backbone network, pointnet2, transformer, dgcnn')
        parser.add_argument('--sampled_num', type=int, default=2048, help='number of sampled 3d points')
        parser.add_argument('--max_dist', type=float, default=-1, help='max distance for clamp')

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['rec']
        self.visual_names = ['input']

        self.model_names = ['E', 'D']
        
        self.netE = networks.define_IMP_encoder(opt.input_nc, opt.output_nc, opt.backbone, opt.is_TAM, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD = networks.define_IMP_decoder(opt.output_nc * 2 + 3, opt.dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_E)
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_D)
            # self.criterionMSELoss = torch.nn.MSELoss().to(self.device)
            self.criterionL1Loss = torch.nn.L1Loss(reduction='none').to(self.device)
        else:
            pass

    def optimize_parameters(self):
        # forward
        self.forward()
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()

        self.compute_loss()
        self.loss.backward()

        self.optimizer_E.step()
        self.optimizer_D.step()

    def set_input(self, input):
        fake_fps = input['FPCfps'].to(self.device)
        # fake_pc = input['FPC'].to(self.device)
        fake_pc = torch.clone(fake_fps)
        fake_label = input['FLabel'].to(self.device)
        real_fps = input['RPCfps'].to(self.device)
        # real_pc = input['RPC'].to(self.device)
        real_pc = torch.clone(real_fps)
        real_label = input['RLabel'].to(self.device)

        self.rec_input = torch.cat((fake_fps, real_fps), dim=0)
        self.rec_target = torch.cat((fake_pc, real_pc), dim=0)

        self.fake_paths = input['FPaths']
        self.real_paths = input['RPaths']

    def forward(self):
        # classification
        if not self.opt.pre_train_decoder:
            cls_code = self.netE(self.cls_input)
            self.pred_cls = self.netC(cls_code)

        # reconstruction
        rec_code = self.netE(self.rec_input)
        B, C = rec_code.shape
        
        sampled_points = torch.rand(B, 3, self.opt.sampled_num).to(self.device)

        # from [0. 1) to [-1, 1)
        sampled_points = sampled_points * 2. - 1.

        # direct naive idea
        self.gt_nearest_distances = nearest_distances(sampled_points, self.rec_target.transpose(2, 1))
        # add gt distance clamp to make distance between original points as 0
        # points_distances = self_nearest_distances(self.rec_target.transpose(2, 1))
        # points_distance = torch.mean(points_distances, dim=1)
        points_distances = self_nearest_distances_K(self.rec_target.transpose(2, 1), k=3)
        k_points_distances = points_distances.topk(k=20, dim=1)[0]
        points_distance = torch.mean(k_points_distances, dim=1)

        points_distance = points_distance.repeat(1, self.opt.sampled_num).reshape(-1, self.opt.sampled_num, 1)
        self.gt_nearest_distances -= points_distance / 2.
        self.gt_nearest_distances = torch.clamp(self.gt_nearest_distances, min=0)


        self.gt_nearest_distances = torch.reshape(self.gt_nearest_distances, (B * self.opt.sampled_num, -1))

        repeat_rec_code = rec_code.view(-1, C, 1).repeat(1, 1, self.opt.sampled_num)
        rec_cat_inputs = torch.cat((sampled_points, repeat_rec_code), dim=1)

        rec_cat_inputs = rec_cat_inputs.transpose(2, 1)
        rec_cat_inputs = torch.reshape(rec_cat_inputs, (B * self.opt.sampled_num, -1))

        self.pred_distances = self.netD(rec_cat_inputs)

    def compute_loss(self):
        # self.loss_rec = self.criterionMSELoss(self.pred_distances, self.gt_nearest_distances)
        if self.opt.max_dist > 0:
            loss_rec = self.criterionL1Loss(torch.clamp(self.pred_distances, max=self.opt.max_dist), torch.clamp(self.gt_nearest_distances, max=self.opt.max_dist))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        else:
            loss_rec = self.criterionL1Loss(self.pred_distances, self.gt_nearest_distances)
        self.loss_rec = loss_rec.sum(-1).mean()

        self.loss = self.loss_rec

    def test(self, input):
        fake_fps = input['FPCfps'].to(self.device)
        fake_label = input['FLabel'].to(self.device)
        real_fps = input['RPCfps'].to(self.device)
        real_label = input['RLabel'].to(self.device)

        self.input = real_fps
        self.cls_gt = real_label
        # self.input = fake_fps
        # self.cls_gt = fake_label
        # self.input = torch.cat((fake_fps, real_fps), dim=0)
        # self.cls_gt = torch.cat((fake_label, real_label), dim=0)

        self.fake_paths = input['FPaths']
        self.real_paths = input['RPaths']

        with torch.no_grad():
            z_code = self.netE(self.input)
            self.pred_cls = self.netC(z_code)

    def compute_error(self):
        _, pred_cls = self.pred_cls.max(1)
        correct = pred_cls.eq(self.cls_gt).sum().item()

        print(correct)

    def generate_dense_point_cloud(self, input, num_steps=10, sample_num=200000, num_points=900000, filter_val=0.009, threshold=0.1):
        start = time.time()

        fake_fps = input['FPCfps'].to(self.device)
        fake_label = input['FLabel'].to(self.device)
        real_pc = input['RPC'].to(self.device)
        real_fps = input['RPCfps'].to(self.device)
        real_label = input['RLabel'].to(self.device)

        rec_input = real_fps
        cls_gt = real_label
        self.paths = input['RPaths']

        # rec_input = fake_fps
        # cls_gt = fake_label.detach().cpu().numpy()
        # self.paths = input['FPaths']

        for param in self.netE.parameters():
            param.requires_grad = False
        for param in self.netD.parameters():
            param.requires_grad = False

        # encoding = self.netE(self.fake_input)
        rec_codes = self.netE(rec_input)
        B, C = rec_codes.shape
        
        for i_batch in range(B):
            print('sample %d in one batch' % (i_batch),  end=' | ')

            input_pc = torch.clone(rec_input[i_batch]).detach().cpu().numpy().reshape(-1, 3)
            rec_code = rec_codes[i_batch]

            samples_np = np.zeros((0, 3))
            samples = torch.rand(1, 3, sample_num).float().to(self.device) * 2. - 1.
            samples.requires_grad = True

            repeat_rec_code = rec_code.view(1, C, 1).repeat(1, 1, sample_num)

            i = 0
            while len(samples_np) < num_points:
                print('iteration', i,  end=' | ')

                for j in range(num_steps):
                    # print('refinement', j,  end=' | ')
                    rec_cat_input = torch.cat((samples, repeat_rec_code), dim=1)
                    rec_cat_input = rec_cat_input.transpose(2, 1)
                    rec_cat_input = torch.reshape(rec_cat_input, (sample_num, -1))

                    # df_pred = torch.clamp(self.netD(rec_cat_input), max=self.opt.max_dist)
                    df_pred = self.netD(rec_cat_input)

                    df_pred.sum().backward()

                    gradient = samples.grad.detach()
                    samples = samples.detach()
                    df_pred = df_pred.detach()
                    rec_input = rec_input.detach()
                    samples = samples - F.normalize(gradient, dim=1) * df_pred.reshape(-1)  # better use Tensor.copy method?
                    samples = samples.detach()
                    samples.requires_grad = True


                # print('finished refinement',  end=' | ')

                if not i == 0:
                    samples_np = np.vstack((samples_np, samples[0, :, df_pred.view(sample_num) < filter_val].transpose(1, 0).detach().cpu().numpy().reshape(-1, 3)))

                samples = samples[:, :, df_pred.view(sample_num) < (threshold / 3)]
                indices = torch.randint(samples.shape[2], (1, sample_num))
                samples = samples[:, :, indices[0]]
                samples += (threshold / 3) * torch.randn(samples.shape).to(self.device)  # 3 sigma rule
                samples = samples.detach()
                samples.requires_grad = True

                i += 1
                print('samples num points: ', samples_np.shape,  end=' | ')

            input_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_pc))
            output_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples_np))            

            # write xyz
            output_path = './rec_output/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            input_pc_path = self.paths[i_batch]
            if type(input_pc_path) == bytes:
                input_pc_path  = input_pc_path.decode('UTF-8')
                path_name = input_pc_path.split('|')[0].split('/')[-1].split('.')[0]
            else:
                path_name = str(input_pc_path)

            label_name = str(cls_gt[i_batch])
            input_pc_path_write = output_path + 'inp_' + path_name + '_' + label_name + '.xyz'
            rec_pc_path_write = output_path + 'rec_' + path_name + '_' + label_name + '.xyz'

            o3d.io.write_point_cloud(input_pc_path_write, input_pc)
            o3d.io.write_point_cloud(rec_pc_path_write, output_pc)

            duration = time.time() - start

            print('finale points num', samples_np.shape, ' time: ', duration)

        return samples_np, duration

    def generate_dense_point_cloud_directly(self, input, num_steps=10, sample_num=400000, num_points=900000, filter_val=0.009, threshold=0.06):
        fps = input['PC'].to(self.device)
        # fake_pc = input['FPC'].to(self.device)
        pc = torch.clone(fps)
        label = input['Label'].to(self.device)

        rec_input = pc
        cls_gt = label.detach().cpu().numpy()
        self.paths = input['Paths']

        # encoding = self.netE(self.fake_input)
        rec_codes = self.netE(rec_input)
        B, C = rec_codes.shape
        
        for i_batch in range(B):
            print('sample %d in one batch' % (i_batch),  end=' | ')

            rec_code = rec_codes[i_batch]

            samples = torch.rand(1, 3, sample_num).float().to(self.device) * 2. - 1.
            # samples = torch.rand(1, 3, sample_num).float().to(self.device)

            # samples_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(torch.clone(samples).transpose(2, 1).detach().cpu().numpy().reshape(-1, 3)))
            # o3d.io.write_point_cloud('./0_smp_.xyz', samples_pc)

            samples.requires_grad = True

            repeat_rec_code = rec_code.view(1, C, 1).repeat(1, 1, sample_num)

            rec_cat_input = torch.cat((samples, repeat_rec_code), dim=1)
            rec_cat_input = rec_cat_input.transpose(2, 1)
            rec_cat_input = torch.reshape(rec_cat_input, (sample_num, -1))

            # # gt
            # df_pred = nearest_distances(samples, rec_targets[i_batch:i_batch + 1].transpose(2, 1))
            # # clamp
            # # points_distances = self_nearest_distances(rec_targets[i_batch:i_batch + 1].transpose(2, 1))
            # points_distances = self_nearest_distances_K(rec_targets[i_batch:i_batch + 1].transpose(2, 1))
            
            # # points_distance = torch.mean(points_distances, dim=1)
            # k_points_distances = points_distances.topk(k=20, dim=1)[0]
            # points_distance = torch.mean(k_points_distances, dim=1)

            # points_distance = points_distance.repeat(1, sample_num).reshape(-1, sample_num, 1)
            # df_pred -= points_distance / 2
            # df_pred = torch.clamp(df_pred, min=0)
            # df_pred = torch.reshape(df_pred, (sample_num, -1))

            df_pred = self.netD(rec_cat_input)
            # # df_pred = torch.sqrt(df_pred)

            samples = samples[:, :, df_pred.view(sample_num) < threshold].transpose(2, 1).detach().cpu().numpy().reshape(-1, 3)

            print('num output points: ', samples.shape,  end=' | ')
            
            # input_pc_o3d = o3d.geometry.PointCloud()
            # input_pc_o3d.points = o3d.utility.Vector3dVector(input_pc)
            output_pc_o3d = o3d.geometry.PointCloud()
            output_pc_o3d.points = o3d.utility.Vector3dVector(samples)

            # write xyz
            output_path = './rec_output/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            input_pc_path = self.paths[i_batch]

            scene_name = input_pc_path.split('/')[-1].split('_')[1]
            frame_name = input_pc_path.split('/')[-1].split('_')[2]
            # graspnet
            label_name = str(cls_gt[i_batch])
            # input_pc_path_write = output_path + 'inp_' + label_name + '_' + scene_name + '_' + frame_name + '.xyz'
            rec_pc_path_write = output_path + 'rec_' + label_name + '_' + scene_name + '_' + frame_name + '.xyz'

            # o3d.io.write_point_cloud(input_pc_path_write, input_pc_o3d)
            o3d.io.write_point_cloud(rec_pc_path_write, output_pc_o3d)

    def get_input_point_cloud(self, input):
        fake_fps = input['FPCfps'].to(self.device)
        # fake_pc = input['FPC'].to(self.device)
        fake_pc = torch.clone(fake_fps)
        fake_label = input['FLabel'].to(self.device)
        real_fps = input['RPCfps'].to(self.device)
        # real_pc = input['RPC'].to(self.device)
        real_pc = torch.clone(real_fps)
        real_label = input['RLabel'].to(self.device)

        # rec_input = real_fps
        # rec_targets = real_pc
        # cls_gt = real_label.detach().cpu().numpy()
        # self.paths = input['RPaths'].detach().cpu().numpy() # for pointda modelnet path
        # # self.paths = input['RPaths']  # for graspnet

        rec_input = fake_fps
        rec_targets = fake_pc
        cls_gt = fake_label.detach().cpu().numpy()
        self.paths = input['FPaths']

        B, _, _ = rec_input.shape
        for i_batch in range(B):
            print('sample %d in one batch' % (i_batch),  end=' | ')

            input_pc = torch.clone(rec_input[i_batch]).detach().cpu().numpy().reshape(-1, 3) # 1024
            # input_pc = torch.clone(rec_targets[i_batch]).detach().cpu().numpy().reshape(-1, 3) # 2048

            input_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_pc))

            # write xyz
            output_path = './rec_output/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            input_pc_path = self.paths[i_batch]
            if type(input_pc_path) == bytes:
                input_pc_path  = input_pc_path.decode('UTF-8')
                path_name = input_pc_path.split('|')[0].split('/')[-1].split('.')[0]
                scene_name = input_pc_path.split('|')[0].split('/')[-4]   # graspnet
            else:
                path_name = str(input_pc_path)

            # pointda-10
            label_name = str(cls_gt[i_batch])
            input_pc_path_write = output_path + 'inp_' + path_name + '_' + label_name + '.xyz'

            # graspnet
            # label_name = str(cls_gt[i_batch])
            # input_pc_path_write = output_path + 'inp_' + scene_name + '_' + path_name + '_' + label_name + '.xyz'

            o3d.io.write_point_cloud(input_pc_path_write, input_pc)