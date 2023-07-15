import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

'''
Networks for point clouds
'''
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # Run on cpu or gpu
    # device = torch.device("cuda:" + str(x.get_device()) if args.cuda else "cpu")

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature

def l2_norm(input, axit=1):
    norm = torch.norm(input, 2, axit, True)
    output = torch.div(input, norm)
    return output

class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                # nn.BatchNorm2d(out_ch),
                # nn.InstanceNorm2d(out_ch),
                # TransNorm2d(out_ch),
                nn.LayerNorm([out_ch, 1024, 20]),
                nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                # nn.BatchNorm2d(out_ch),
                # nn.InstanceNorm2d(out_ch),
                # TransNorm2d(out_ch),
                nn.LayerNorm([out_ch, 1024, 20]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                # nn.BatchNorm1d(out_ch),
                nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        x = l2_norm(x, 1)
        x = self.fc(x)
        return x

class transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """

    def __init__(self, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out

        activation = 'leakyrelu'
        bias = False

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation=activation, bias=bias)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation=activation, bias=bias)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation=activation, bias=bias)
        self.fc1 = fc_layer(1024, 512, activation=activation, bias=bias, bn=True)
        self.fc2 = fc_layer(512, 256, activation=activation, bn=True)
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        # device = torch.device("cuda:" + str(x.get_device()))

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        
        x = x.max(dim=-1, keepdim=False)[0]
        x = torch.unsqueeze(x, dim=3)

        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

class DGCNN(nn.Module):
    def __init__(self, input_nc, output_nc, K=20):
        super(DGCNN, self).__init__()
        self.k = K

        # self.input_transform_net = transform_net(input_nc, 3)

        self.conv1 = conv_2d(input_nc * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.conv5 = nn.Conv1d(num_f_prev, output_nc, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(output_nc)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        # x0 = get_graph_feature(x, k=self.k)  # x0: [b, 6, 1024, 20]
        # align to a canonical space (e.g., apply rotation such that all inputs will have the same rotation)
        # transformd_x0 = self.input_transform_net(x0)  # transformd_x0: [3, 3]
        # x = torch.matmul(transformd_x0, x)

        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        x = get_graph_feature(x, k=self.k)  # x: [b, 6, 1024, 20]
        # process point and inflate it from 6 to e.g., 64
        x = self.conv1(x)  # x: [b, 64, 1024, 20]
        # per each feature (from e.g., 64) take the max value from the representative vectors
        # Conceptually this means taking the neighbor that gives the highest feature value.
        # returns a tensor of size e.g., (batch_size, 64, #points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv5(x_cat)  # [b, 1024, 1024]
        x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
        x1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # x5 = F.leaky_relu(self.bn5(x), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        # x5_pool = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        # x = x5_pool

        return x

# Regresssion
class PC_Regression(nn.Module):
    def __init__(self, input_nc, dropout=0.5):
        super(PC_Regression, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(input_nc, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        pred = self.regression(x)
        return pred

class Pct_simply(nn.Module):
    def __init__(self, input_nc, mid_nc, output_nc):
        super(Pct_simply, self).__init__()
        self.conv1 = nn.Conv1d(input_nc, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)

        self.pt_last = Point_Transformer_Last()

        self.conv_fuse = nn.Sequential(
                                    nn.Conv1d(1280, mid_nc, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(mid_nc, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_nc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, is_embedding):
        batch_size, _, _ = x.size()
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # B, channel = 256, N

        y = self.pt_last(x)
        x = torch.cat([x, y], dim=1)
        x = self.conv_fuse(x)

        if is_embedding:
            return x
        
        else:
            x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.linear3(x)

            return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

def define_IMP_encoder(input_nc, output_nc, netE, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if netE == 'dgcnn':
        net = DGCNN(input_nc, output_nc)
    elif netE == 'pct_simply':
        net = Pct_simply(input_nc, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netE)

    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=True)

def define_IMP_decoder(input_nc, dropout, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = PC_Regression(input_nc=input_nc, dropout=dropout)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=True)