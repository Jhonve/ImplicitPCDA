import itertools
from statistics import mode
import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from measurement.mmd_numpy_scipy import *
from measurement.wasserstein_distance import SinkhornDistance

k_KLDivLoss = nn.KLDivLoss(reduction='batchmean')
k_shikhorn = SinkhornDistance(eps=0.1, max_iter=100)

k_source_feats_path = 'kinect_latent_feas_method.npy'
k_source_labels_path = 'kinect_labels_method.npy'

k_target_feats_path = 'real_latent_feas_defrec.npy'
k_target_labels_path = 'real_labels_defrec.npy'

def js_divergence(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    if get_softmax:
        p_output = F.softmax(p_output, dim=1)
        q_output = F.softmax(q_output, dim=1)
    log_mean_output = ((p_output + q_output )/2).log()
    return (k_KLDivLoss(log_mean_output, p_output) + k_KLDivLoss(log_mean_output, q_output))/2

def get_distances_cross_domains():
    source_feats = np.load(k_source_feats_path)
    source_labels = np.load(k_source_labels_path)

    target_feats = np.load(k_target_feats_path)
    target_labels = np.load(k_target_labels_path)

    num_source = source_feats.shape[0]
    num_target = target_feats.shape[0]

    if (num_source > num_target):
        repeat_array = np.ones(num_target)
        repeat_array[0:(num_source - num_target)] = 2
        target_feats = np.repeat(target_feats, repeat_array.astype(np.int16), axis=0)
    elif (num_target > num_source):
        repeat_array = np.ones(num_source)
        repeat_array = repeat_array * 2
        repeat_array[0:(num_target - 2 * num_source)] = 3
        source_feats = np.repeat(source_feats, repeat_array.astype(np.int16), axis=0)

    print('Number of source features %d and number of target features %d'%(num_source, num_target))

    mmd_liner_distance = mmd_linear(source_feats, target_feats)
    mmd_rbf_distance = mmd_rbf(source_feats, target_feats)
    mmd_poly_distance = mmd_poly(source_feats, target_feats)
    print('MMD distance linear %f, rbf %f, poly %f'%(mmd_liner_distance, mmd_rbf_distance, mmd_poly_distance))

    wasserstein_distance, _, _ = k_shikhorn(torch.tensor(source_feats), torch.tensor(target_feats))
    print('Wasserstein distance %f'%(wasserstein_distance))

    source_feats_kl = F.softmax(torch.tensor(source_feats), dim=1)
    target_feats_kl = F.softmax(torch.tensor(target_feats), dim=1)
    kl_div = k_KLDivLoss(target_feats_kl.log(), source_feats_kl)
    print('Kullback-Leibler divergence %f'%(kl_div))

    js_div = js_divergence(torch.tensor(source_feats), torch.tensor(target_feats))
    print('Jense-Shannon divergence %f'%(js_div))

def get_distances_cross_classes(mode='source', num_classes=10):
    # source domain
    if mode == 'source':
        feats_path = k_source_feats_path
        labels_path = k_source_labels_path
    elif mode == 'target':
        feats_path = k_target_feats_path
        labels_path = k_target_labels_path
    else:
        print('wrong mode')
        exit(0)

    feats = np.load(feats_path)
    labels = np.load(labels_path)

    distances = np.zeros((10, 10, 7)) # calculate kl distances twice
    for i_class in range(num_classes):
        i_class_indices = (labels == i_class)
        i_class_feats_all = feats[i_class_indices]
        for j_class in range(i_class, num_classes):
            j_class_indices = (labels == j_class)
            j_class_feats_all = feats[j_class_indices]

            num_i = i_class_feats_all.shape[0]
            num_j = j_class_feats_all.shape[0]

            # if (num_i > num_j):
            #     i_class_feats = i_class_feats_all
            #     repeat_array = np.ones(num_j)
            #     repeat_array[0:(num_i - num_j)] = 2
            #     j_class_feats = np.repeat(j_class_feats_all, repeat_array.astype(np.int16), axis=0)
            # elif (num_j > num_i):
            #     repeat_array = np.ones(num_i)
            #     repeat_array[0:(num_j - num_i)] = 2
            #     i_class_feats = np.repeat(i_class_feats_all, repeat_array.astype(np.int16), axis=0)
            #     j_class_feats = j_class_feats_all
            # else:
            #     i_class_feats = i_class_feats_all
            #     j_class_feats = j_class_feats_all

            if (num_i > num_j):
                i_class_feats = i_class_feats_all[0:num_j]
                j_class_feats = j_class_feats_all
            elif (num_j > num_i):
                i_class_feats = i_class_feats_all
                j_class_feats = j_class_feats_all[0:num_i]
            else:
                i_class_feats = i_class_feats_all
                j_class_feats = j_class_feats_all

            distances[i_class, j_class, 0] = mmd_linear(i_class_feats, j_class_feats)
            distances[i_class, j_class, 1] = mmd_rbf(i_class_feats, j_class_feats)
            distances[i_class, j_class, 2] = mmd_poly(i_class_feats, j_class_feats)

            wasserstein_distance, _, _ = k_shikhorn(torch.tensor(i_class_feats), torch.tensor(j_class_feats))
            distances[i_class, j_class, 3] = wasserstein_distance

            i_class_feats_kl = F.softmax(torch.tensor(i_class_feats), dim=1)
            j_class_feats_kl = F.softmax(torch.tensor(j_class_feats), dim=1)
            distances[i_class, j_class, 4] = k_KLDivLoss(i_class_feats_kl.log(), j_class_feats_kl)
            distances[i_class, j_class, 5] = k_KLDivLoss(j_class_feats_kl.log(), i_class_feats_kl)

            distances[i_class, j_class, 6] = js_divergence(torch.tensor(i_class_feats), torch.tensor(j_class_feats))

    return distances

k_pointda_types = ['Bathtub', 'Bed', 'Bookshelf', 'Cabinet', 'Chair', 'Lamp', 'Monitor', 'Plant', 'Sofa', 'Table',]
def draw_matrix(mat, classes, title='Confusion matrix', cmap=plt.cm.Blues, name='dis'):
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 14,
            }

    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)

    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = mat.max() / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, format(mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if mat[i, j] > thresh else "black")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.ylabel('Labels', font)
    plt.xlabel('Labels', font)
    plt.savefig(name + '.png', dpi=300)

def draw_distances(distances):
    mat_0 = (distances[:, :, 0] / np.max(distances[:, :, 0])) + \
         (distances[:, :, 2].transpose() / np.max(distances[:, :, 2]))
    draw_matrix(mat_0, k_pointda_types, cmap=plt.cm.YlGnBu, name='dis_0')

    mat_1 = (distances[:, :, 3] / np.max(distances[:, :, 3])) + \
         (distances[:, :, 6].transpose() / np.max(distances[:, :, 6]))
    draw_matrix(mat_1, k_pointda_types, cmap=plt.cm.YlGnBu, name='dis_1')

def get_distances_cross_domains_classes(num_classes=10):
    source_feats = np.load(k_source_feats_path)
    source_labels = np.load(k_source_labels_path)

    target_feats = np.load(k_target_feats_path)
    target_labels = np.load(k_target_labels_path)

    distances = np.zeros((10, 7)) # calculate kl distances twice
    for i_class in range(num_classes):
        i_source_class_indices = (source_labels == i_class)
        i_source_class_feats_all = source_feats[i_source_class_indices]

        i_target_class_indices = (target_labels == i_class)
        i_target_class_feats_all = target_feats[i_target_class_indices]

        num_source = i_source_class_feats_all.shape[0]
        num_target = i_target_class_feats_all.shape[0]

        if (num_source > num_target):
            i_source_class_feats = i_source_class_feats_all[0:num_target]
            i_target_class_feats = i_target_class_feats_all
        elif (num_target > num_source):
            i_source_class_feats = i_source_class_feats_all
            i_target_class_feats = i_target_class_feats_all[0:num_source]
        else:
            i_source_class_feats = i_source_class_feats_all
            i_target_class_feats = i_target_class_feats_all

        distances[i_class, 0] = mmd_linear(i_source_class_feats, i_target_class_feats)
        distances[i_class, 1] = mmd_rbf(i_source_class_feats, i_target_class_feats)
        distances[i_class, 2] = mmd_poly(i_source_class_feats, i_target_class_feats)

        wasserstein_distance, _, _ = k_shikhorn(torch.tensor(i_source_class_feats), torch.tensor(i_target_class_feats))
        distances[i_class, 3] = wasserstein_distance

        i_class_feats_kl = F.softmax(torch.tensor(i_source_class_feats), dim=1)
        j_class_feats_kl = F.softmax(torch.tensor(i_target_class_feats), dim=1)
        distances[i_class, 4] = k_KLDivLoss(i_class_feats_kl.log(), j_class_feats_kl)
        distances[i_class, 5] = k_KLDivLoss(j_class_feats_kl.log(), i_class_feats_kl)

        distances[i_class, 6] = js_divergence(torch.tensor(i_source_class_feats), torch.tensor(i_target_class_feats))

    return distances

def draw_distances_cross_domains(distances_c, distances_s, distances_t):
    mat_0 = (distances_s[:, :, 0] / np.max(distances_s[:, :, 0])) + \
         (distances_t[:, :, 0].transpose() / np.max(distances_t[:, :, 0]))

    for i in range(distances_c.shape[0]):
        mat_0[i, i] = distances_c[i, 0] / 16. / 16. / 2.

    mat_0 = np.clip(mat_0, 0., 1.)

    draw_matrix(mat_0, k_pointda_types, cmap=plt.cm.YlGnBu, name='dis_d_0')

    # * 1.5 + \ for no adapt
    mat_2 = (distances_s[:, :, 2] / np.max(distances_s[:, :, 2])) + \
         (distances_t[:, :, 2].transpose() / np.max(distances_t[:, :, 2]))

    # / 16. / 32. / 64. / 32. / 2. for no adapt
    # / 16. / 32. / 64. / 32. / 8. for no gast
    for i in range(distances_c.shape[0]):
        mat_2[i, i] = distances_c[i, 2] / 16.

    mat_2 = np.clip(mat_2, 0., 1.)

    draw_matrix(mat_2, k_pointda_types, cmap=plt.cm.YlGnBu, name='dis_d_2')

'''
if __name__ == '__main__':
    get_distances_cross_domains()
    distances_s = get_distances_cross_classes(mode='source')
    distances_t = get_distances_cross_classes(mode='target')
    draw_distances(distances_s)

    distances_c = get_distances_cross_domains_classes()
    draw_distances_cross_domains(distances_c, distances_s, distances_t)
'''