import argparse
import os

import models

import torch

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='Debug', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # dataset parameters
        parser.add_argument('--model', type=str, default='implicit', help='chooses which model to use.')
        parser.add_argument('--dataset', type=str, default='GraspNet', choices=['GraspNet', 'PointDAN', 'PointDANSynthetic'], help='choose whihc dataset to train.')
        parser.add_argument('--datapath_prepared', type=bool, default=False, help='Wherether to generate depth path file.')
        parser.add_argument('--num_class', type=int, default=10, help='...')
        parser.add_argument('--points_num', type=int, default=1024, help='...')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--rotation_augmentation', type=bool, default=True, help='whether to use dataset augmentation by random rotation')
        # datapath parameters 
        parser.add_argument('--datapath_fake', type=str, default='../Dataset/PointDANData/modelnet/', help='Path to Synthetic dataset')
        parser.add_argument('--datapath_real', type=str, default='../Dataset/PointDANData/scannet/', help='Path to Real-scan dataset')
        parser.add_argument('--datapath_h5', type=str, default='./datasets/', help='Path to h5 file for saving data pathes')
        parser.add_argument('--datapath_file_fake', type=str, default='./datasets/PointDANDataPathFakeModelNet.h5', help='H5 file saved clean data path')
        # parser.add_argument('--datapath_file_real', type=str, default='./datasets/PointDANDataPathFakeShapeNet.h5', help='H5 file saved noise data path')

        # network parameters
        parser.add_argument('--input_nc', type=int, default=3, help='...')
        parser.add_argument('--output_nc', type=int, default=512, help='...')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout of classification network')
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--is_sync_norm', type=bool, default=True, help='...')
        # point cloud parameters
        parser.add_argument('--camera_mode', type=str, default='realsense', help='kinect or realsense')
        parser.add_argument('--depth_scale', type=int, default=1000, help='...')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.is_train)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / [phase]_opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt