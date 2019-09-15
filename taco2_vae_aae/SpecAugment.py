import numpy as np
import random
import torch
from torch.autograd import Variable
from layers import TacotronSTFT
from audio_processing import griffin_lim, mel_denormalize
from hparams import create_hparams
import os
import time
from scipy.io.wavfile import write
import argparse
from utils import load_wav_to_torch
import matplotlib.pylab as plt

'''
def time_warping_numpy(image):
    # get width and heigt
    num_rows = image.shape[0]
    spec_len = image.shape[1]
    W = spec_len // 4

    # get control point location
    rp = np.random.randint(W, spec_len - W)
    source_control_point_locations = ([[[0, 0, 1], [0, num_rows - 1, 1], [rp, 0, 1], [rp, num_rows - 1, 1]],
                                       [[rp + 1, num_rows - 1, 1], [rp + 1, 0, 1], [spec_len - 1, num_rows - 1, 1],
                                        [spec_len - 1, 0, 1]]])
    w = np.random.randint(-W, W)
    dest_control_point_locations = ([[[0, 0, 1], [0, num_rows - 1, 1], [rp + w, 0, 1], [rp + w, num_rows - 1, 1]],
                                     [[rp + w + 1, num_rows - 1, 1], [rp + w + 1, 0, 1],
                                      [spec_len - 1, num_rows - 1, 1], [spec_len - 1, 0, 1]]])

    warp_line = rp + w
    # calculate inverse matrix
    trans_left = np.matmul(np.linalg.pinv(dest_control_point_locations[0]), source_control_point_locations[0])
    trans_right = np.matmul(np.linalg.pinv(dest_control_point_locations[1]), source_control_point_locations[1])

    # generate empty warped_image
    warped_image = np.zeros((num_rows, spec_len))
    warped_point = np.ones((spec_len*num_rows, 3))

    generate_warp_point = True
    i = 0
    a = 0
    b = 0
    while generate_warp_point:
      warped_point[i][0] = a
      warped_point[i][1] = b
      i += 1
      b += 1
      if b == (num_rows):
        a += 1
        b = 0
      if a == (spec_len):
        generate_warp_point = False

    origin_point_left = np.matmul(warped_point[:warp_line*num_rows, :], trans_left)[:,:2]
    origin_point_right = np.matmul(warped_point[warp_line*num_rows:, :], trans_right)[:,:2]
    origin_point = np.concatenate([origin_point_left, origin_point_right])

    k = 0
    for i in range(spec_len):
      for j in range(num_rows):
        warped_image[j, i] = bilinear_interpolation_numpy(image, origin_point[k], spec_len, num_rows, i, warp_line, rp)
        k += 1
    print("random point: ", rp, "   w: ", w)
    return Variable(torch.from_numpy(warped_image))


def bilinear_interpolation_numpy(image, origin_point, spec_len, num_rows, i, warp_line, rp):
    if i <= warp_line:
        hor_upper = np.clip(int(np.ceil(origin_point[0])), 0, rp)
        hor_lower = np.clip(int(np.trunc(origin_point[0])), 0, rp)
    else:
        hor_upper = np.clip(int(np.ceil(origin_point[0])), rp + 1, spec_len - 1)
        hor_lower = np.clip(int(np.trunc(origin_point[0])), rp + 1, spec_len - 1)
    ver = np.clip(int(np.round(origin_point[1])), 0, num_rows - 1)

    w1 = np.absolute(origin_point[0] - (hor_lower))
    w2 = np.absolute((hor_upper) - origin_point[0])

    p = (w1 / (w1 + w2))
    q = (w2 / (w1 + w2))
    if np.isnan(p):
        value = image[ver][int(origin_point[0])]
    else:
        value = p * image[ver][hor_upper] + q * image[ver][hor_lower]
    return value
'''
def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_data(data, index=1, output_dir="./test", figsize=(16, 4)):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    #
    axes.imshow(data, aspect='auto', origin='lower',
                        interpolation='none')
    plt.savefig(os.path.join(output_dir, 'sentence_{}.png'.format(index)))

def time_warping(image, w_length_rate = 0.08):
    # get width and height
    num_rows = image.size()[0]
    spec_len = image.size()[1]
    W = spec_len // 4

    # get control point location
    rp = np.random.randint(W, spec_len - W)

    w = np.random.randint(-int(np.round(spec_len * w_length_rate)), int(np.round(spec_len * w_length_rate)))
    warp_line = rp + w

    source_control_point_locations = torch.as_tensor(
        [[[0, 0, 1], [0, num_rows - 1, 1], [rp, 0, 1], [rp, num_rows - 1, 1]],
         [[rp, num_rows - 1, 1], [rp, 0, 1], [spec_len - 1, num_rows - 1, 1],
          [spec_len - 1, 0, 1]]]).double()
    dest_control_point_locations = torch.as_tensor(
        [[[0, 0, 1], [0, num_rows - 1, 1], [rp + w, 0, 1], [rp + w, num_rows - 1, 1]],
         [[rp + w, num_rows - 1, 1], [rp + w, 0, 1], [spec_len - 1, num_rows - 1, 1],
          [spec_len - 1, 0, 1]]]).double()

    # get inverse matrix
    trans_left = torch.matmul(torch.from_numpy(np.linalg.pinv(dest_control_point_locations[0])),
                              source_control_point_locations[0]).float()
    trans_right = torch.matmul(torch.from_numpy(np.linalg.pinv(dest_control_point_locations[1])),
                               source_control_point_locations[1]).float()

    # wapred image index [spec_len*num_rows, 3]
    warped_point = torch.cat(
        [(torch.zeros(num_rows)).float().unsqueeze(1), torch.arange(0, num_rows).unsqueeze(1).float(),
         (torch.ones(num_rows)).float().unsqueeze(1)], dim=1)
    for i in range(spec_len - 1):
        temp = torch.cat(
            [(torch.ones(num_rows) * (i + 1)).float().unsqueeze(1), torch.arange(0, num_rows).unsqueeze(1).float(),
             (torch.ones(num_rows)).float().unsqueeze(1)], dim=1)
        warped_point = torch.cat([warped_point, temp], dim=0)

    # calculate origin location
    origin_point_left = torch.matmul(warped_point[:warp_line * num_rows, :], trans_left)[:, :2]
    origin_point_right = torch.matmul(warped_point[warp_line * num_rows:, :], trans_right)[:, :2]
    origin_point = torch.cat([origin_point_left, origin_point_right])

    ##Bilinear interpolation

    # cliping horizontal index matrix
    hor_upper_left = torch.clamp((torch.ceil(origin_point[:warp_line * num_rows, 0])), 0, rp).int()
    hor_lower_left = torch.clamp((torch.trunc(origin_point[:warp_line * num_rows, 0])), 0, rp).int()
    hor_upper_right = torch.clamp((torch.ceil(origin_point[warp_line * num_rows:, 0])), rp, spec_len - 1).int()
    hor_lower_right = torch.clamp((torch.trunc(origin_point[warp_line * num_rows:, 0])), rp, spec_len - 1).int()
    hor_upper = torch.cat([hor_upper_left, hor_upper_right])
    hor_lower = torch.cat([hor_lower_left, hor_lower_right])

    # get variables
    w1 = torch.abs(origin_point[:, 0] - (hor_lower.float()))
    w2 = torch.abs((hor_upper.float()) - origin_point[:, 0])
    w1[w1 == 0] = 0.5
    w2[w2 == 0] = 0.5
    p_ = (w1 / (w1 + w2)).float()
    q_ = (w2 / (w1 + w2)).float()

    # transpose index matrix(upper, lower, p, q) [spec_len, num_rows] -> [num_rows, spec_len]
    upper = hor_upper[0:num_rows].unsqueeze(0)
    lower = hor_lower[0:num_rows].unsqueeze(0)
    p = p_[0:num_rows].unsqueeze(0)
    q = q_[0:num_rows].unsqueeze(0)
    for i in range(spec_len - 1):
        temp_1 = hor_upper[num_rows * (i + 1):num_rows * (i + 2)]
        upper = torch.cat([upper, temp_1.unsqueeze(0)])
        temp_2 = hor_lower[num_rows * (i + 1): num_rows * (i + 2)]
        lower = torch.cat([lower, temp_2.unsqueeze(0)])
        temp_p = p_[num_rows * (i + 1): num_rows * (i + 2)]
        p = torch.cat([p, temp_p.unsqueeze(0)])
        temp_q = q_[num_rows * (i + 1): num_rows * (i + 2)]
        q = torch.cat([q, temp_q.unsqueeze(0)])
    upper = torch.transpose(upper, 0, 1).contiguous().view(1, -1).squeeze()
    lower = torch.transpose(lower, 0, 1).contiguous().view(1, -1).squeeze()
    p = torch.transpose(p, 0, 1).contiguous().view(1, -1).squeeze()
    q = torch.transpose(q, 0, 1).contiguous().view(1, -1).squeeze()

    # generate empty warped_image
    warped_image = torch.zeros((num_rows, spec_len))

    # get image value by index matrix
    upper_value = torch.index_select(image[0], 0, upper[0:spec_len].long()).squeeze()
    lower_value = torch.index_select(image[0].unsqueeze(1), 0, lower[0:spec_len].long()).squeeze()
    for i in range(num_rows - 1):
        temp_1 = torch.index_select(image[i + 1].unsqueeze(1), 0,
                                    upper[spec_len * (i + 1):spec_len * (i + 2)].long()).squeeze()
        upper_value = torch.cat([upper_value, temp_1])
        temp_2 = torch.index_select(image[i + 1].unsqueeze(1), 0,
                                    lower[spec_len * (i + 1):spec_len * (i + 2)].long()).squeeze()
        lower_value = torch.cat([lower_value, temp_2])

        # calculate warped image value
    value = torch.mul(p, upper_value) + torch.mul(q, lower_value)

    # get warped image
    for i in range(num_rows):
        warped_image[i] = value[spec_len * i:spec_len * (i + 1)]

    return warped_image

def warping_range(rate = 0.08):
    return np.random.uniform(-rate, rate, 1)

def volumeDown(rate = 0.16):
    return np.random.uniform(1 - rate, 1.0, 1)

def time_length_adjustment(image, w):
    # get width and height
    num_rows = image.size()[0]
    spec_len = image.size()[1]

    # get control point location
    #w = int(np.round(spec_len * warping_range))
    # if spec_len * warping_range <= 0.5 or spec_len == 2:
    #     return image
    # else:
    #w = np.random.randint(-int(np.round(spec_len * warping_range)), int(np.round(spec_len * warping_range)))

    source_control_point_locations = torch.as_tensor(
        [[[0, 0, 1], [0, num_rows - 1, 1], [spec_len, 0, 1], [spec_len, num_rows - 1, 1]]]).double()
    dest_control_point_locations = torch.as_tensor(
        [[0, 0, 1], [0, num_rows - 1, 1], [spec_len + w, 0, 1], [spec_len + w, num_rows - 1, 1]]).double()
    # get inverse matrix
    trans = torch.matmul(torch.from_numpy(np.linalg.pinv(dest_control_point_locations)),
                         source_control_point_locations).float().squeeze()

    # wapred image index [(spec_len+w)*num_rows, 3]
    warped_point = torch.cat(
        [(torch.zeros(num_rows)).float().unsqueeze(1), torch.arange(0, num_rows).unsqueeze(1).float(),
         (torch.ones(num_rows)).float().unsqueeze(1)], dim=1)
    for i in range(spec_len - 1 + w):
        temp = torch.cat(
            [(torch.ones(num_rows) * (i + 1)).float().unsqueeze(1), torch.arange(0, num_rows).unsqueeze(1).float(),
             (torch.ones(num_rows)).float().unsqueeze(1)], dim=1)
        warped_point = torch.cat([warped_point, temp], dim=0)
    # calculate origin location
    origin_point = torch.matmul(warped_point, trans)[:, :2]

    ##Bilinear interpolation

    # cliping horizontal index matrix
    hor_upper = torch.clamp((torch.ceil(origin_point[:, 0])), 0, spec_len).int()
    hor_lower = torch.clamp((torch.trunc(origin_point[:, 0])), 0, spec_len).int()
    # get variables
    w1 = torch.abs(origin_point[:, 0] - (hor_lower.float()))
    w2 = torch.abs((hor_upper.float()) - origin_point[:, 0])
    w1[w1 == 0] = 0.5
    w2[w2 == 0] = 0.5
    p_ = (w1 / (w1 + w2)).float()
    q_ = (w2 / (w1 + w2)).float()

    # transpose index matrix(upper, lower, p, q) [spec_len, num_rows] -> [num_rows, spec_len]
    upper = hor_upper[0:num_rows].unsqueeze(0)
    lower = hor_lower[0:num_rows].unsqueeze(0)
    p = p_[0:num_rows].unsqueeze(0)
    q = q_[0:num_rows].unsqueeze(0)
    for i in range(spec_len - 1 + w):
        temp_1 = hor_upper[num_rows * (i + 1):num_rows * (i + 2)]
        upper = torch.cat([upper, temp_1.unsqueeze(0)])
        temp_2 = hor_lower[num_rows * (i + 1): num_rows * (i + 2)]
        lower = torch.cat([lower, temp_2.unsqueeze(0)])
        temp_p = p_[num_rows * (i + 1): num_rows * (i + 2)]
        p = torch.cat([p, temp_p.unsqueeze(0)])
        temp_q = q_[num_rows * (i + 1): num_rows * (i + 2)]
        q = torch.cat([q, temp_q.unsqueeze(0)])
    upper = torch.transpose(upper, 0, 1).contiguous().view(1, -1).squeeze()
    lower = torch.transpose(lower, 0, 1).contiguous().view(1, -1).squeeze()
    p = torch.transpose(p, 0, 1).contiguous().view(1, -1).squeeze()
    q = torch.transpose(q, 0, 1).contiguous().view(1, -1).squeeze()
    image = torch.cat([image, image[:, :50]], dim=1)
    # generate empty warped_image
    warped_image = torch.zeros((num_rows, spec_len + w))
    # get image value by index matrix
    upper_value = torch.index_select(image[0], 0, upper[0:spec_len + w].long()).squeeze()
    lower_value = torch.index_select(image[0].unsqueeze(1), 0, lower[0:spec_len + w].long()).squeeze()
    for i in range(num_rows - 1):
        temp_1 = torch.index_select(image[i + 1].unsqueeze(1), 0,
                                    upper[(spec_len + w) * (i + 1):(spec_len + w) * (i + 2)].long()).squeeze()
        upper_value = torch.cat([upper_value, temp_1])
        temp_2 = torch.index_select(image[i + 1].unsqueeze(1), 0,
                                    lower[(spec_len + w) * (i + 1):(spec_len + w) * (i + 2)].long()).squeeze()
        lower_value = torch.cat([lower_value, temp_2])

    # calculate warped image value
    value = torch.mul(p, upper_value) + torch.mul(q, lower_value)

    # get warped image
    for i in range(num_rows):
        warped_image[i] = value[(spec_len + w) * i:(spec_len + w) * (i + 1)]

    return warped_image

def local_time_length_adjustment(image, location, warping_range):
    """
    :param image: melspectorgram [mel dim, time dim]
    :param location: (start time, end time)
    :param warping_range: adjustment rate for time length
    :return: adjusted image
    """

    part1 = image[:, :location[0]]
    part2 = image[:, location[0]:location[1]]
    part3 = image[:, location[1]:, :]
    part2_ = time_length_adjustment(part2, warping_range)

    image_ = torch.concat((part1,part2_,part3), 1)

    return image_


def local_random_time_warping(image, w_length_rate=0.4):
    spec_len = image.size()[1]
    warp_point = 0
    block_size = 0
    location = [0, 0]
    part_ = []
    image_ = torch.FloatTensor()

    while warp_point < spec_len:
        block_size = np.random.randint(15, 30)
        location[0] = warp_point
        location[1] = warp_point + block_size

        if location[1] <= spec_len:
            w = np.random.randint(-int(np.round(block_size * w_length_rate)), int(np.round(block_size * w_length_rate)))
            part = image[:, location[0]:location[1]]
            part_.append(time_length_adjustment(part, w))
        else:
            part = image[:, location[0]:]
            part_.append(part)

        warp_point = location[1]

    for i in range(0, len(part_)):
        image_ = torch.cat((image_, part_[i]), 1)
    # spec_len = image.size()[1]
    # block_size = 20
    # warp_num = int(spec_len / block_size)
    # warp_point = []
    # location = []
    # part = []
    # part_ = []
    # image_ = torch.FloatTensor()
    #
    # for i in range(0, warp_num):
    #     warp_point.append(block_size * i)
    #     location.append(warp_point[i])
    #
    # for i in range(0, warp_num-1):
    #     part.append(image[:, location[i]:location[i+1]])
    # part.append(image[:, location[warp_num-1]:])
    #
    # for i in range(0, warp_num):
    #     part_.append(time_length_adjustment(part[i], w_length_rate))
    #
    # for i in range(0, warp_num):
    #     image_ = torch.cat((image_, part_[i]), 1)

    return image_

def local_random_freq_warping(image, warp_range=4):

    spec_len = image.size()[1]
    warp_point = 0
    block_size = 0
    location = [0, 0]
    part_ = []
    image_ = torch.FloatTensor()

    while warp_point < spec_len:
        block_size = np.random.randint(15, 30)
        location[0] = warp_point
        location[1] = warp_point + block_size

        w = np.random.randint(-warp_range, warp_range)
        rp = np.random.randint(20, 60)

        if location[1] <= spec_len:
            part = image[:, location[0]:location[1]]
            part_.append(freq_warping(part, rp, w))
        else:
            part = image[:, location[0]:]
            part_.append(freq_warping(part, rp, w))

        warp_point = location[1]

    for i in range(0, len(part_)):
        image_ = torch.cat((image_, part_[i]), 1)

    #warp_num = int(spec_len / block_size)
    #part = []
    #warp_point = []
    # for i in range(0, warp_num):
    #     warp_point.append(block_size * i)
    #     location.append(warp_point[i])
    #
    # for i in range(0, warp_num - 1):
    #     part.append(image[:, location[i]:location[i + 1]])
    # part.append(image[:, location[warp_num - 1]:])
    #
    # for i in range(0, warp_num):
    #     part_.append(freq_warping(part[i], warp_range))
    #     image_ = torch.cat((image_, part_[i]), 1)

    return image_


def time_mask(image, T=4, num_masks=2, replace_with_zero=True, max_abs_mel_value = 4.0):
    cloned = image.clone()
    len_spectro = cloned.size()[1]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[:, t_zero:mask_end] = -max_abs_mel_value

        else:
            cloned[:, t_zero:mask_end] = cloned.mean()
    return cloned


def freq_mask(image, F=3, num_masks=2, replace_with_zero=True, max_abs_mel_value = 4.0):
    cloned = image.clone()
    num_mel_channels = cloned.size()[0]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[f_zero:mask_end, :] = -max_abs_mel_value
        else:
            cloned[f_zero:mask_end, :] = cloned.mean()

    return cloned

def value_adjustment(image, scale): #scale in (0,1)
    #print(scale)
    image_ = (image - image.min()) * torch.tensor(scale,dtype=torch.float) + image.min()
    #print(image_.shape, image_.min(), image_.max())
    return image_

def freq_warping(image, rp, w):
    # get width and heigt
    num_rows = image.size()[0]

    spec_len = image.size()[1]

    # get control point location
    #rp = np.random.randint(20, 60)
    source_control_point_locations = torch.as_tensor(
        [[[0, 0, 1], [0, rp, 1], [spec_len, 0, 1], [spec_len, rp, 1]],
         [[0, rp, 1], [0, num_rows - 1, 1], [spec_len, rp, 1], [spec_len, num_rows - 1, 1]]]).double()

    #w = np.random.randint(-w_length, w_length)

    dest_control_point_locations = torch.as_tensor(
        [[[0, 0, 1], [0, rp + w, 1], [spec_len, 0, 1], [spec_len, rp + w, 1]],
         [[0, rp + w, 1], [0, num_rows - 1, 1], [spec_len, rp + w, 1], [spec_len, num_rows - 1, 1]]]).double()
    warp_line = rp + w

    # get inverse matrix
    trans_lower = torch.matmul(torch.from_numpy(np.linalg.pinv(dest_control_point_locations[0])),
                               source_control_point_locations[0]).float()
    trans_upper = torch.matmul(torch.from_numpy(np.linalg.pinv(dest_control_point_locations[1])),
                               source_control_point_locations[1]).float()

    # generate empty warped_image
    warped_image = torch.zeros((num_rows, spec_len))

    # warped image index [num_rows, 3]
    warped_point = torch.zeros((num_rows, 3), dtype=torch.float32)
    for i in range(num_rows):
        warped_point[i, 0] = 0
        warped_point[i, 1] = i
        warped_point[i, 2] = 1

    # calculate origin location
    origin_point_lower = torch.matmul(warped_point[:warp_line, :], trans_lower)[:, :2]
    origin_point_upper = torch.matmul(warped_point[warp_line:, :], trans_upper)[:, :2]
    origin_point = torch.cat([origin_point_lower, origin_point_upper])

    ##Bilinear interpolation
    hor_upper_left = torch.clamp(torch.ceil(origin_point[:warp_line, 1]), 0, rp).int()
    hor_lower_left = torch.clamp(torch.trunc(origin_point[:warp_line, 1]), 0, rp).int()
    hor_upper_right = torch.clamp(torch.ceil(origin_point[warp_line:, 1]), rp, num_rows - 1).int()
    hor_lower_right = torch.clamp(torch.trunc(origin_point[warp_line:, 1]), rp, num_rows - 1).int()
    upper = torch.cat([hor_upper_left, hor_upper_right])
    lower = torch.cat([hor_lower_left, hor_lower_right])

    # get variables
    w1 = torch.abs(origin_point[:, 0] - (lower.float()))
    w2 = torch.abs((upper.float()) - origin_point[:, 0])
    w1[w1 == 0] = 0.5
    w2[w2 == 0] = 0.5
    p = (w1 / (w1 + w2)).float()
    p = p.repeat(spec_len)
    q = (w2 / (w1 + w2)).float()
    q = q.repeat(spec_len)

    # get image value by index matrix
    upper_value = torch.index_select(image[:, 0], 0, upper.long()).squeeze()
    lower_value = torch.index_select(image[:, 0], 0, lower.long()).squeeze()
    for i in range(spec_len - 1):
        temp_1 = torch.index_select(image[:, i + 1].unsqueeze(1), 0,
                                    upper.long()).squeeze()
        upper_value = torch.cat([upper_value, temp_1])
        temp_2 = torch.index_select(image[:, i + 1].unsqueeze(1), 0,
                                    lower.long()).squeeze()
        lower_value = torch.cat([lower_value, temp_2])

    # calculate warped image value``
    value = torch.mul(p, upper_value) + torch.mul(q, lower_value)
    # get warped image
    for i in range(spec_len):
        warped_image[:, i] = value[num_rows * i:num_rows * (i + 1)]

    return warped_image

def local_freq_warping(image, location):
    """
    :param image: melspectorgram [mel dim, time dim]
    :param location: (start time, end time)
    :param warping_range: adjustment rate for time length
    :return: adjusted image
    """

    part1 = image[:, :location[0]]
    part2 = image[:, location[0]:location[1]]
    part3 = image[:, location[1]:]
    part2_ = time_length_adjustment(part2)

    image_ = torch.concat((part1,part2_,part3), 1)

    return image_

def test(hparams, mel, output_path="test.wav", ref_level_db = 20, magnitude_power=1.5):
    taco_stft = TacotronSTFT(hparams)
    stime = time.time()
    mel_decompress = mel_denormalize(mel).unsqueeze(0)
    mel_decompress = taco_stft.spectral_de_normalize(mel_decompress + ref_level_db) ** (1 / magnitude_power)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :]),
                           taco_stft.stft_fn, 60)
    waveform = waveform[0].data.cpu().numpy()
    waveform = waveform / abs(waveform).max() * 0.99 * 2**15
    waveform = waveform.astype(dtype=np.int16)
    dec_time = time.time() - stime
    len_audio = float(len(waveform)) / float(hparams.sampling_rate)
    str = "audio length: {:.2f} sec,  mel_to_wave time: {:.2f}".format(len_audio, dec_time)
    print(str)
    write(os.path.join(output_path), hparams.sampling_rate, waveform)

def get_mel(stft, filename, hparams, silence_mel_padding):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, hparams.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    pad = -hparams.max_abs_mel_value*torch.ones((1, 80, silence_mel_padding))
    melspec = torch.cat((melspec, pad), dim=2)
    return melspec

def prameter_experiment():

    tmc = [2, 4, 6, 8, 10, 12, 14, 16] # time masking chunk
    fmc = [2, 4, 6, 8, 10, 12, 14, 16] # frequency maksing chunk
    # tmn = [(1,8), (2,4), (4,2), (8,1)] # time masking chunk number
    # fmn = [(1,6), (2,3), (3,2), (6,1)] # frequency masking chunk number
    twlr = [2, 4, 6, 8, 10, 12, 14, 16] # time warping length ratio
    fwl = [2, 4, 6, 8, 10, 12, 14, 16] #
    tlar = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    flar = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3] # time length adjust ratio
    var = [2, 4, 8, 16, 32, 64]
    lrtw = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]# [0.2, 0.4, 0.8, 1]
    try_nums = np.arange(1,2)
    hparams = create_hparams()
    stft = TacotronSTFT(hparams)

    # taking filelist about validation data
    with open('./filelists/meta_val.txt', encoding='utf-8-sig') as f:
        files = [x.strip().split('|')[0] for x in f.readlines()]

    # file to mel
    mels = []
    for x in files:
        mel = get_mel(stft, x, hparams, 0).squeeze(0)
        mels.append(mel)
        test(hparams,mel,"./test/test.wav")
        #plot_data(mel, 100)
    # average length of mel
    avg_len = np.average([mel.size(1) for mel in mels])
    print(avg_len)

    # griffin lim
    # os.makedirs('gl', exist_ok=True)
    # for i, mel in enumerate(mels):
    #     path = 'gl' + '/{}.wav'.format(i)
    #     test(hparams, mel, path)

    for try_num in try_nums:

        output_dir = 'try{}'.format(try_num)
        os.makedirs(output_dir, exist_ok=True)

        # making a directory for time warping length rate

        flar_path = output_dir + '/FLAR'
        lrtw_path = output_dir + '/LRTW'

        ## warping part

        # time warping length rate
        # ex = 0
        # for r in lrtw:
        #     dir = lrtw_path + '/{}'.format(r)
        #     os.makedirs(dir, exist_ok=True)
        #     for i, mel in enumerate(mels):
        #         path = dir + '/{}.wav'.format(i)
        #         mel_ = local_random_time_warping(mel, 0.4)
        #         plot_data(mel_, ex)
        #         ex += 1
        #         test(hparams, mel_, path)

        print("--------------------------------------------")
        # for r in twlr:
        #     dir = twlr_path + '/{}'.format(r)
        #     os.makedirs(dir, exist_ok=True)
        #     for i, mel in enumerate(mels):
        #         path = dir + '/{}.wav'.format(i)
        #         mel_ = time_warping(mel, r/100.0)
        #         test(hparams, mel_, path)
        #
        #
        # # frequency warping length
        # for l in fwl:
        #     dir = fwl_path + '/{}'.format(l)
        #     os.makedirs(dir, exist_ok=True)
        #     for i, mel in enumerate(mels):
        #         path = dir + '/{}.wav'.format(i)
        #         mel_ = freq_warping(mel,l)
        #         test(hparams, mel_, path)
        #
        # # time length adjustment rate
        # for r in tlar:
        #     dir = tlar_path + '/{}'.format(r)
        #     os.makedirs(dir, exist_ok=True)
        #     for i, mel in enumerate(mels):
        #         path = dir + '/{}.wav'.format(i)
        #         mel_ = local_random_time_warping(mel, warping_range(r/100.0))
        #         print(mel_.size())
        #         test(hparams, mel_, path)
        #
        ex = 0
        for r in flar:
            dir = flar_path #+ '/{}'.format(r)
            os.makedirs(dir, exist_ok=True)
            for i, mel in enumerate(mels):
                path = dir + '/{}.wav'.format(ex)
                mel_ = local_random_freq_warping(mel, r)
                plot_data(mel_, ex)
                ex += 1
                test(hparams, mel_, path)

        ## masking part

        # Adjusting mel value
        # for r in var:
        #     dir = var_path + '/{}'.format(r)
        #     os.makedirs(dir, exist_ok=True)
        #     for i, mel in enumerate(mels):
        #         path = dir + '/{}.wav'.format(i)
        #         mel_ = value_adjustment(mel, volumeDown(r / 100))
        #         test(hparams, mel_, path)

        # masking number

        # # time masking number
        # for n in tmn:
        #     dir = tmn_path + '/{}'.format(n)
        #     os.makedirs(dir, exist_ok=True)
        #     for i, mel in enumerate(mels):
        #         path = dir + '/{}.wav'.format(i)
        #         mel_ = time_mask(mel, n[0], n[1])
        #         test(hparams, mel_, path)
        #
        # # frequency masking number
        # for n in fmn:
        #     dir = fmn_path + '/{}'.format(n)
        #     os.makedirs(dir, exist_ok=True)
        #     for i, mel in enumerate(mels):
        #         path = dir + '/{}.wav'.format(i)
        #         mel_ = freq_mask(mel, n[0], n[1])
        #         test(hparams, mel_, path)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-o', '--output_directory', type=str, default='',
#                         help='directory to save wave and fig')
#     parser.add_argument('-m', '--mel_path_file', type=str, default=None,
#                         required=True, help='sentence path')
#     parser.add_argument('--hparams', type=str,
#                         required=False, help='comma separated name=value pairs')
#
#     args = parser.parse_args()
#     hparams = create_hparams(args.hparams)
#     hparams.sampling_rate = 22050
#     hparams.filter_length = 1024
#     hparams.hop_length = 256
#     hparams.win_length = 1024
#
#     torch.backends.cudnn.enabled = hparams.cudnn_enabled
#     torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
#
#     mel = torch.from_numpy(np.load(args.mel_path_file)).squeeze(0)
#     mel_ = freq_warping(mel, 2)
#     #mel_ = mel
#     # mel = time_mask(mel, T=2, num_masks=4, replace_with_zero=True)
#     # mel = freq_mask(mel, F=2, num_masks=4, replace_with_zero=True)
#     #test(hparams, mel_, args.output_directory)
#     plot_spectrogram_to_numpy(mel_.squeeze(0).numpy())

if __name__ == '__main__':
    prameter_experiment()