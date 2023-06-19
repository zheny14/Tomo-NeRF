import random
import numpy as np
import torch
import learn2learn as l2l
from torch import nn, optim
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import cv2
import os
import glob
import zipfile
import functools
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
# from google.colab.patches import cv2_imshow
from PIL import Image
from skimage import io

# from positional_encodings import PositionalEncodingPermute3D
##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary
import torchvision.models as models

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb

import copy
import random
import time
import tifffile
from tifffile import imsave
"Parameters"


"start testing sample"
acc_epoch= 0
"Overall testing samples"
num_testing_sample=5


"parameter for crop image"
"the minimum crop image dimension"
h = 100
w = 150
# h = 60
# w =90

"Load and order data path"
# Data path
"Artificial image"
DATA_PATH = "Testing_artificial_samples_for_testing"
"Real image"
#DATA_PATH = "Real_images_for_testing"

#DATA_PATH = "F:\particle_data\data\data_9_output"
#DATA_PATH = "F:\particle_data\Post_Processing_Packed_Bed\Post Processing\data_output_real_2"

nData = len(os.listdir(DATA_PATH))

# Sort png files
all_png = []
import os, re, glob

files = glob.glob(os.path.join(DATA_PATH, '*.png'))
files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

for infile in files:
    all_png.append(infile)
    # print(infile)

# Sort tiff files
all_tiff = []

import os, re, glob

files = glob.glob(os.path.join(DATA_PATH, '*.tiff'))
files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

for infile in files:
    all_tiff.append(infile)
    all_tiff = all_tiff

# each task: 1 tiff and 12 png data path
sample_img_path = []
png_per_sample = []
tiff_per_sample = []
img_per_sample = []  # 12 png + one tiff at the end
all_sample_img = []

#######
h_cutoff_down = 100
h_cutoff_up = 250
w_cutoff_down = 150
w_cutoff_up = 154

"""change here to len(all_tiff)"""
for i in range(len(all_tiff)-acc_epoch):  # len(all_tiff) ->20 try 20 samples first
    # eliminate the image set that outside the size range
    fixed_projection = io.imread(all_png[(acc_epoch+i)*12])  # pick the radiography at 0 deg
    h_img = fixed_projection.shape[0]
    w_img = fixed_projection.shape[1]
    if (h_cutoff_down<=h_img<=h_cutoff_up) & (w_cutoff_down<=w_img<=w_cutoff_up):
        img_per_sample = all_png[(acc_epoch+i) * 12:(acc_epoch+i + 1) * 12:1]
        tiff_per_sample = all_tiff[acc_epoch+i]
        img_per_sample.append(tiff_per_sample)
        all_sample_img.append(img_per_sample)
    else:
        continue

main_list_iterator=iter(all_sample_img)
"################################################################"
"""Input Parameters"""
" model parameters"
weight_factor = 25  # add to MSE loss
ResNet_Output_DIM = 1024
n_positional_encoding = 16
INPUT_DIM = 3
OUTPUT_DIM = 2

# BATCH_SIZE =1024
# IMAGE_BATCH_SIZE=16
BATCH_SIZE =4096
IMAGE_BATCH_SIZE=16


"parameters for rotation"
# rotate the x-ray beam (rotate voxel location)
# tiff img rotate counter clock-wise
# is equivalent to
# x-ray beam location rotate clock-wise
"Adjust for different number of view reconstruction"
rotate_angle_counter_clock = np.arange(15, 180, 15)
#rotate_angle_counter_clock = np.arange(30, 180, 30)
#rotate_angle_counter_clock = 90*np.ones(1)

rotate_angle_clock = -rotate_angle_counter_clock  # because artificial radiography is created by rotating tiff counterclockwise, and make projection. So the x-ray beam is considered rotate clockwise
center = (w / 2, w / 2)
n_projection = len(rotate_angle_counter_clock) + 1

"parameters for radiography imaging"

# Imaging properties
actual_resolution_um = 26.33 # um/pixel
mu_l_p = 1/4064.03  # linear attenuation coefficient of particle [1/um] @30keV
mu_l_quartz = 1/5835.76  # linear attenuation coefficient of quartz [1/um]
eff_detector = 0.45  # quantum efficiency

# Determine quartz wall thickness with x
quartz_thickness = np.zeros(w)
#real
# OR = 4.45 / 2  # outside radius [mm]
# IR = 3.95 / 2  # inside radius [mm]
#artificial
OR = 5.95 / 2  # outside radius [mm]
IR = 3.95 / 2  # inside radius [mm]


x_quartz = np.linspace(-IR, IR, w, endpoint=True)
for i in range(w):
    quartz_thickness[i] = 1000 * 2 * (
            np.sqrt([OR ** 2 - x_quartz[i] ** 2]) - np.sqrt([IR ** 2 - x_quartz[i] ** 2]))  # [um]

quartz_thickness = torch.from_numpy(quartz_thickness)

"Adjust the parameters for "
"Properties for adjust pixel grayscale (strech histogram to [0,1])"

# real sample
# avg_origin =0.252
# sigma_origin = 0.0386
# n_std = 3.3
# mu_l_p = 1/4064#4064.03  # linear attenuation coefficient of particle [1/um] @30keV
# mu_l_quartz = 1/4835.76#5835.76  # linear attenuation coefficient of quartz [1/um]

# # artificial sample
mu_l_p = 1/4064.03  # linear attenuation coefficient of particle [1/um] @30keV
mu_l_quartz = 1/5835.76  # linear attenuation coefficient of quartz [1/um]
avg_origin =0.20782
sigma_origin = 0.02994
n_std = 3.5


min_grayscale_bound = avg_origin - n_std * sigma_origin
max_grayscale_bound = avg_origin + n_std * sigma_origin  # average and sigma of original photo (before adjusted are found in matlab e.g. average the value for several test)

"################################################################"
"Functions"
def crop_img(img, cropx, cropy):
    # define a img function to horizontally center crop and vertically crop from one edge
    startx = (img.shape[1] - cropx) // 2
    endx = startx + cropx
    endy = cropy
    return img[0:endy, startx:endx]
    # return img[0:endy, startx:endx]

def batch_rotate(original_location, center, rotation_angle):
    # point rotate function
    def rotate(p, origin, degrees):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    location_rotate_3d = np.copy(original_location)
    location_rotate_3d[:, :2] = rotate(original_location[:, :2], center, rotation_angle)
    return location_rotate_3d




def occupancy_for_all_tiffs(tiff_input, center, rotation_angle_list):
    # input is original tiff, rotate center, and counterclock-wise rotate angle
    # output is the all the occupancy in lists [each list is a beam]
    occupancy_datainput = []
    original_tiffarray = tiff_input
    tiffarray0 = tiff_input.copy()
    "uncomment it later"
    for j in range(len(rotation_angle_list)):  # attach the 11 rotate tiffs with the original tiff
        tiffarray_rotate = np.zeros((w, w, h))
        original_slice = np.array(original_tiffarray)  # original slice before rotate
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation_angle_list[j], scale=1)
        rotated_image = cv2.warpAffine(src=original_slice, M=rotate_matrix, dsize=(w, w))
        tiffarray_rotate = rotated_image
        tiffarray0 = np.concatenate((tiffarray0, tiffarray_rotate), axis=2)
    all_tiffarray = tiffarray0.astype(np.double)
    for k in range(n_projection):
        one_tiff = np.copy(all_tiffarray[:, :, k * h:(k + 1) * h])
        occupancy = one_tiff[
            voxel_location_np[:, 0].astype(int), voxel_location_np[:, 1].astype(int), voxel_location_np[:, 2].astype(
                int)]
        occupancy = occupancy.reshape(B, N, 1)
        occupancy = np.rint(occupancy)
        occupancy_datainput.append(torch.Tensor(occupancy).type(torch.float32))
    occupancy_datainput_each_sample = torch.cat(occupancy_datainput, axis=0)
    occupancy_datainput_each_sample = list(occupancy_datainput_each_sample)
    return occupancy_datainput_each_sample

# plt.imshow(fixed_projection[:,:], cmap='gray')
# plt.show()
# plt.axis('off')

def cnn_image_loader(iterator):
    fixed_projection = io.imread(iterator[0])
    print(iterator[0])
    "real image"
    fixed_projection = np.flip(fixed_projection,
                              0)  #comment it for real image
    fixed_projection = crop_img(fixed_projection, w, h)/ 255

    #add 90deg
    fixed_projection_90= io.imread(iterator[6])
    fixed_projection_90 = np.flip(fixed_projection_90,
                               0)   #comment it for real image
    fixed_projection_90 = crop_img(fixed_projection_90, w, h)/ 255

    fixed_projection = np.concatenate((fixed_projection, fixed_projection_90), axis=0)

    mean,std=fixed_projection.mean(),fixed_projection.std()
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), transforms.Resize(size=[224,224])])
    fixed_projection = preprocess(fixed_projection)
    fixed_projection = fixed_projection.type(torch.float32)
    #fixed_projection = torch.stack([fixed_projection], dim=0)
    fixed_projection = torch.stack([fixed_projection, fixed_projection, fixed_projection, fixed_projection], dim=0)
    return fixed_projection

"Functions for training/testing"
def calculate_accuracy(y_pred, y):
    y_pred = y_pred.detach().clone()
    #y_pred = y_pred.argmax(1, keepdim=True)
    correct = y_pred.eq(y).sum()
    acc = correct.float() / y.shape[0]
    return acc
    #return ((y_pred>0.5).float()==y.squeeze()).float().mean()


def reconstruc_radiography(y_pred, original_projection_location_x):  # add x[0]
    # produce grayscale
    # y_pred = y_pred.detach().clone()
    # y_pred = y_pred.argmax(1, keepdim=True)
    #y_pred = y_pred[:, 1].float()
    y_pred = torch.reshape(y_pred, (-1, w))
    # x_location_in_batch = torch.mean(location_xz, axis=1)[:, 0]
    Grayscale_beam = eff_detector * torch.exp(
        -mu_l_p * actual_resolution_um * torch.sum(y_pred, dim=1) - mu_l_quartz * quartz_thickness[
            original_projection_location_x.long()])
    Grayscale_beam_adjusted = (Grayscale_beam - min_grayscale_bound) * (1 - 0) / (
            max_grayscale_bound - min_grayscale_bound) + 0
    return Grayscale_beam_adjusted

"###################################################################"
"Prepare normalized voxel location lists (12 projections + positional encoding)-> for query set -> same for all query samples"
"Prepare normalized voxel location lists (1 projection + positional encoding)-> for support set -> same for all support samples"

"voxel_input_pe_(support/query) -> normalize + 1 projection/12 projections + positional encoding"
"voxel_location -> original location without rotation"
voxel_location_np = []  # original numpy
voxel_location_norm_np = []  # normalized numpy

voxel_location = []  # original
voxel_location_norm = []  # normalized
x_location_one_projection = []

for i in range(h):
    for j in range(w):
        # original location
        voxel_location_row = np.ones((3, w))
        voxel_location_row[0, :] = (j * voxel_location_row[0, :])
        voxel_location_row[1, :] = (np.arange(w))
        voxel_location_row[2, :] = (i * voxel_location_row[2, :])
        voxel_location_row = np.transpose(voxel_location_row)
        voxel_location_np.append(voxel_location_row)

        # voxel x location -> for calculating the glass grey shade
        voxel_x_location = j
        x_location_one_projection.append(voxel_x_location)

        # normalized location
        voxel_location_row_norm = np.ones((3, w))
        voxel_location_row_norm[0, :] = (j * voxel_location_row_norm[0, :]) / (w - 1)
        voxel_location_row_norm[1, :] = (np.arange(w)) / (w - 1)
        voxel_location_row_norm[2, :] = (i * voxel_location_row_norm[2, :]) / (h - 1)
        voxel_location_row_norm = np.transpose(voxel_location_row_norm)
        voxel_location_norm_np.append(voxel_location_row_norm)

x_location = np.tile(np.array(x_location_one_projection),
                     n_projection)  # concatenate all the original x location for all projections
x_location = list(torch.tensor(x_location, dtype=torch.int32))

voxel_location_np0 = np.array(voxel_location_np)
voxel_location = torch.tensor(voxel_location_np0, dtype=torch.int32)

voxel_location_norm_np0 = np.array(voxel_location_norm_np)
voxel_location_norm = torch.tensor(voxel_location_norm_np0, dtype=torch.float32)

voxel_location_norm_np = np.array(voxel_location_norm_np)
voxel_location_np = np.array(voxel_location_np)
B, N, C = voxel_location_norm_np.shape
voxel_location_np = voxel_location_np.reshape(-1, 3)  # the original location before rotate
voxel_location_norm_np = voxel_location_norm_np.reshape(-1, 3)

voxel_rotate_location_norm = []
voxel_rotate_location = []
for k in range(len(rotate_angle_clock)):
    voxel_rotate_3d = batch_rotate(np.copy(voxel_location_np), center, rotate_angle_clock[k])
    voxel_rotate_3d = voxel_rotate_3d.reshape(B, N, C)
    # normalize here
    voxel_rotate_3d = np.rint(voxel_rotate_3d)
    voxel_rotate_location.append(torch.Tensor(voxel_rotate_3d).type(torch.int32))

    voxel_rotate_3d[:, :, 0] = voxel_rotate_3d[:, :, 0] / (w - 1)
    voxel_rotate_3d[:, :, 1] = voxel_rotate_3d[:, :, 1] / (w - 1)
    voxel_rotate_3d[:, :, 2] = voxel_rotate_3d[:, :, 2] / (h - 1)
    voxel_rotate_location_norm.append(torch.Tensor(voxel_rotate_3d).type(torch.float32))

location_buffer = []
location_buffer = torch.cat(voxel_rotate_location_norm, 0)
voxel_input = []
voxel_input = torch.cat((voxel_location_norm, location_buffer), axis=0)

voxel_input_origin = []
location_buffer_origin = []
location_buffer_origin = torch.cat(voxel_rotate_location, 0)
voxel_input_origin = torch.cat((voxel_location, location_buffer_origin), axis=0)

voxel_label=torch.arange(len(voxel_input_origin)).long()
#
"positional encoding"
# def p_e_3d(x, n):
#     x_new = np.copy(x)
#     for i in range(n):
#         x_p = np.sin(x * (2 ^ i))
#         x_new = np.concatenate((x_new, x_p), 1)
#     return x_new
#
# n_positional_encoding = 16
# voxel_np = np.array(voxel_input_origin)
# voxel_np_pe = []
#
# for i in range(len(voxel_np)):
#     voxel_np_pe.append(p_e_3d(voxel_np[i][:], n_positional_encoding))  # change into 16
#     if (i % 10000) == 0:
#         print(i)
#
# voxel_input_origin = torch.tensor(np.array(voxel_np_pe))

# the 3d dimension will be expanded to (n_positional_encoding+1)*3
# voxel_np = np.array(voxel_input)
# voxel_np_pe = []
# for i in range(len(voxel_np)):
#     voxel_np_pe.append(p_e_3d(voxel_np[i][:], n_positional_encoding))  # change into 16
#     if (i % 10000) == 0:
#         print(i)
#
# voxel_np_pe_0 = np.array(voxel_np_pe)
# voxel_input_pe = torch.tensor(voxel_np_pe_0, dtype=torch.float32)
# voxel_input_pe_query = list(voxel_input_pe)
# voxel_input_pe_support =voxel_input_pe[:h*w,:,:] # for support set (one shot), only select 1st projection
# voxel_input_pe_support=list(voxel_input_pe_support)
# voxel_location = list(voxel_location) #original for  reconstruct the 3d image



def per_task_dataset_loader(img_list_iterator):
    # load tomography
    Tiff_im = Image.open(img_list_iterator[12])
    rows, columns = np.shape(Tiff_im)  # find the original shape
    #numSlices = Tiff_im.n_frames  # find the original shape
    tiffarray = np.zeros((w, w, h))
    for i in range(h):
        Tiff_im.seek(i)
        tiffarray[:, :, i] = np.array(Tiff_im)[int((rows - w) / 2):int(rows - (rows - w) / 2),
                             int((rows - w) / 2):int(rows - (rows - w) / 2)]
    occupancy_label = occupancy_for_all_tiffs(tiffarray, center,
                                              rotate_angle_counter_clock)  # list of occupancy label for all rotate angles

    # load radiography label
    all_projection_png = []
    for j in range(n_projection):
        projection_img = io.imread(img_list_iterator[j]) / 255  # find the first figure
        projection_img = np.flip(projection_img,
                                 0)  #comment it for real image
        projection_img = crop_img(projection_img, w, h)
        projection_img = projection_img.flatten()
        if j == 0:
            all_projection_png = projection_img
        else:
            all_projection_png = np.concatenate((all_projection_png, projection_img), axis=0)

        all_radiograph = list(torch.tensor(all_projection_png.copy(), dtype=torch.float32))

    per_task_dataset = [voxel_input_origin, occupancy_label, all_radiograph,
                     x_location]  # voxel location of x-ray beam imput, tomography 3d label
    per_task_dataset = [[voxel_input_query, three_d_label_query, two_d_label_query, x_location_query] for (voxel_input_query, three_d_label_query, two_d_label_query, x_location_query) in
                     zip(per_task_dataset[0], per_task_dataset[1], per_task_dataset[2], per_task_dataset[3])]

    return per_task_dataset


#ConvTranspose3d
# inputs = torch.from_numpy(np.random.randn(BATCH_SIZE, 1, 18, 18,11).astype(np.float32))
# outputs = nn.ConvTranspose3d(BATCH_SIZE, 1, 8, stride=4, padding=2)(inputs)
# print(outputs.shape) #torch.Size([16, 1, 72, 72, 44])

#bilinear sampling

#the grid below works
d1 = torch.linspace(-1, 1, w)
d2=torch.linspace(-1,1,h)
meshx, meshy,meshz = torch.meshgrid((d1, d1,d2))
grid = torch.stack((meshx, meshy,meshz), 3).unsqueeze(0)
#grid = grid.unsqueeze(0).repeat(BATCH_SIZE,1,1,1,1) # add batch dim


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=2 if downsample else 1,
                               padding=1)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1)
        self.shortcut = nn.Sequential()

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=512):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]

        else:
            filters = [64, 64, 128, 256, 512]
            #filters = [32, 32, 64, 128, 256]

            #filters = [64, 256, 512, 1024, 2048]

            #filters = [32, 32, 64, 64, 64]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module('conv2_%d' % (i + 1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
            self.layer2.add_module('conv3_%d' % (i + 1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i + 1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d' % (i + 1,), resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        #self.fc = torch.nn.Linear(filters[4], outputs)
        #self.fc = torch.nn.Linear(filters[3], outputs)


    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        #input = self.fc(input)

        return input

#resnet18_model = ResNet(1, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=512)
resnet18_model = ResNet(1, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=512) #resnet34

#resnet18_model = ResNet(1, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=2048)
#resnet18_model = ResNet(1, ResBottleneckBlock, [3, 8, 36, 3], useBottleneck=True, outputs=2048)

#[3, 4, 6, 3] 50
#[3, 8, 36, 3], 152


#output = torch.nn.functional.grid_sample(inp, grid,mode='bilinear',padding_mode='zeros',align_corners=None)
class Net_upsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=resnet18_model
        #self.convt3d_1=nn.ConvTranspose3d(8, 32, 4, stride=2,padding=0)# out = 20,12
        self.convt3d_1=nn.ConvTranspose3d(8, 64, 4, stride=2,padding=0)# out = 20,12

        self.conv_1=nn.Conv3d(64,64,1,stride=1,padding=0)
        self.convt3d_2=nn.ConvTranspose3d(64, 32, 4, stride=2,padding=(0,0,2))# out = 20,12
        self.conv_2=nn.Conv3d(32,32,1,stride=1,padding=0)
        self.convt3d_3=nn.ConvTranspose3d(32, 16, 4, stride=2,padding=(0,0,2))# out = 20,12
        self.conv_3=nn.Conv3d(16,16,1,stride=1,padding=0)
        self.convt3d_4=nn.ConvTranspose3d(16, 1, 4, stride=2,padding=(0,0,2))# out = 20,12
        self.bn0=nn.BatchNorm3d(8)
        self.bn1=nn.BatchNorm3d(64)
        self.bn_c_1=nn.BatchNorm3d(64)
        self.bn2=nn.BatchNorm3d(32)
        self.bn_c_2=nn.BatchNorm3d(32)
        self.bn3=nn.BatchNorm3d(16)
        self.bn_c_3=nn.BatchNorm3d(16)
    def forward(self,x,image):
        y =self.cnn(image)
        y=y.view(y.shape[0],8, 4,4,4)
        # y=F.relu(self.input_fc_1(y))
        # y = F.relu(self.input_fc_2(y)).view(y.shape[0],64, 3,3,2)
        # y=self.input_fc_2(y).view(y.shape[0],32, 4,4,4)
        y=F.relu(self.bn0(y))
        y = F.relu(self.bn1(self.convt3d_1(y)))
        y = F.relu(self.bn_c_1(self.conv_1(y)))
        y = F.relu(self.bn2(self.convt3d_2(y)))
        y = F.relu(self.bn_c_2(self.conv_2(y)))
        y = F.relu(self.bn3(self.convt3d_3(y)))
        y = F.relu(self.bn_c_3(self.conv_3(y)))
        # y = F.relu(self.bn4(self.convt3d_4(y)))
        # y = F.relu(self.bn_c_4(self.conv_4(y)))
        #y = F.relu(self.bn5(self.convt3d_5(y)))

        y = F.sigmoid(self.convt3d_4(y))
        if torch.Tensor.dim(x)==3:
            x=x.unsqueeze(0)
            grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
            y = F.grid_sample(y, grid.to(device), mode='bilinear', padding_mode='zeros').squeeze(0)
        else:
            grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
            y = F.grid_sample(y, grid.to(device), mode='bilinear', padding_mode='zeros').squeeze()

        x=x.view(x.shape[0], -1, x.shape[-1]) # reshape to [image batch size, w*voxel_batch_size, feature dim =3]
        x = x[:, :, 0:3]
        y_pred=[]
        for i in range(x.shape[0]):
            voxel_x = x[i, :, 0]
            voxel_y = x[i, :, 1]
            voxel_z = x[i, :, 2]
            y_pred_per_batch = torch.zeros(x.shape[1], dtype=torch.float32).to(device)
            idx_inside_bound = (voxel_x >= 0) & (voxel_x < w) & (voxel_y >= 0) & (voxel_y < w)
            y_pred_per_batch[idx_inside_bound] = y[i][
                voxel_x[idx_inside_bound].long(), voxel_y[idx_inside_bound].long(), voxel_z[idx_inside_bound].long()]
            y_pred.append(y_pred_per_batch)
        y_pred=torch.cat(y_pred, dim=0)
        y_pred = y_pred.view(len(y_pred), 1)
        y_pred=torch.cat((1 - y_pred, y_pred), 1) #return two
        return y_pred


# #load pretrained model
resnet18=Net_upsampling().cnn
# model_weight_path="./resnet18-5c106cde.pth" #download pretrained model
# ## resnett 34
# model_weight_path="./resnet34-333f7ec4.pth" #download pretrained model
#
# #resnet50
# model_weight_path="./resnet50-19c8e357.pth" #download pretrained model
#
# assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# pre_state_dict = torch.load(model_weight_path)
# print("original model", pre_state_dict.keys())
# new_state_dict = {}
# for k, v in resnet18.state_dict().items():
#     print("new model", k)
#     if k in pre_state_dict.keys() and k!= 'conv1.weight':
#         new_state_dict[k] = pre_state_dict[k]
# resnet18.load_state_dict(new_state_dict, False)

model=Net_upsampling()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=0.0008)
#criterion_1 = nn.CrossEntropyLoss()
criterion_1 = nn.CrossEntropyLoss()
criterion_2 = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
#model=model.to(device)
criterion_1 = criterion_1.to(device)
criterion_2 = criterion_2.to(device)
quartz_thickness=quartz_thickness.to(device)
"##############################################"




" load saved model"
model.load_state_dict(torch.load('Trained_and_optimized_artificial_sample_six_view.pth'))


model.eval()


"reconstruct the image"
def reconstruc_radiography_validate(y_pred, original_projection_location_x):  # add x[0]
    # produce grayscale
    y_pred = y_pred.detach().clone()
    #y_pred = y_pred.argmax(1, keepdim=True)
    batch_n = int(y_pred.shape[0]/w)

    #y_pred = y_pred[:,1].float()

    #y_pred = torch.reshape(y_pred, (batch_n, w))

    y_pred = torch.reshape(y_pred, (batch_n, w))
    #x_location_in_batch = torch.mean(location_xz, axis=1)[:, 0]
    Grayscale_beam = eff_detector * torch.exp(
        -mu_l_p * actual_resolution_um * torch.sum(y_pred, dim=1) - mu_l_quartz * quartz_thickness[
            original_projection_location_x.long() ])
    Grayscale_beam_adjusted = (Grayscale_beam - min_grayscale_bound) * (1 - 0) / (
                max_grayscale_bound - min_grayscale_bound) + 0
    #pdb.set_trace()

    return Grayscale_beam_adjusted


def get_predictions(model, iterator, image, device):
    model.eval()

    images = []
    labels = []
    pixel = []
    real_pixel = []
    epoch_acc = 0
    # probs = []

    with torch.no_grad():

        # for (x, _,_,_) in iterator:
        for (x, y, z, t) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            # y = y.to(device) #actual results
            y = y.to(device).reshape(-1, 1)
            z = z.to(device)
            t = t.to(device)

            y_pred= model(x,image)
            #y_pred= model(x)
            y_pred = y_pred.argmax(1, keepdim=True)

            acc = calculate_accuracy(y_pred, y)
            #y_pred_radio = torch.unsqueeze(y_pred[:, 1].float(), 1)
            radiography_grey_value_pred = reconstruc_radiography_validate(y_pred, t)
            radiography_grey_value_real = reconstruc_radiography_validate(y, t)
            # pdb.set_trace()
            #_pred= (y_pred > 0.5).float()

            # y_prob = F.softmax(y_pred, dim=-1)
            images.append(x.cpu())
            labels.append(y_pred.cpu())
            pixel.append(radiography_grey_value_pred.cpu())
            real_pixel.append(radiography_grey_value_real.cpu())
            epoch_acc += acc.item()
            # probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    pixel = torch.cat(pixel, dim=0)
    real_pixel = torch.cat(real_pixel, dim=0)
    # probs = torch.cat(probs, dim=0)

    return images, labels, pixel, real_pixel,epoch_acc / len(iterator)

quartz_thickness=quartz_thickness.to(device)


def testing_iterator_set(num_training_sample):
    "return set of data and image iterators for random shuffle"
    data_iterator_set=[]
    image_iterator_set=[]
    for i in range(num_training_sample):
        main_data_buffer = next(main_list_iterator)
        print(main_data_buffer[0],i)
        data_input = per_task_dataset_loader(main_data_buffer)
        data_iterator = data.DataLoader(data_input, batch_size=BATCH_SIZE, drop_last=True)
        data_iterator_set.append(data_iterator)
        image_input = cnn_image_loader(main_data_buffer)
        image_iterator_set.append(image_input)
    return data_iterator_set, image_iterator_set

def batch_order_set_testing(num_of_test,iterator_set,image_batch_size): #tps is task per step
    #overall_iterator_len=len(iterator_set['input_data_0'])*num_of_train
    single_list=np.arange(num_of_test)
    all_list=np.tile(single_list, len(iterator_set[0]))
    all_list = torch.tensor(all_list, dtype=torch.float32).long()
    batch_order_iterator=data.DataLoader(all_list, batch_size=image_batch_size,drop_last=True)
    # if use the same order
    return batch_order_iterator


def batch_data_preparation(iterator_set,image_iterator_set, image_batch_size,batch_order):
    "shuffle the order"
    image_batch=image_iterator_set[batch_order[0]]
    x_set,y_set,z_set,t_set=iterator_set[f'input_data_{batch_order[0]}'].next()
    x_set,y_set=x_set.unsqueeze(0),y_set.unsqueeze(0)
    for i in range(image_batch_size-1):
        x, y, z, t = iterator_set[f'input_data_{batch_order[i+1]}'].next()
        image_batch=torch.cat((image_batch,image_iterator_set[batch_order[i+1]]),dim=0)
        x_set,y_set,z_set,t_set=torch.cat((x_set,x.unsqueeze(0)),dim=0),torch.cat((y_set,y.unsqueeze(0)),dim=0),torch.cat((z_set,z),dim=0),torch.cat((t_set,t),dim=0)
    return x_set, y_set, z_set,t_set, image_batch


"Validate the accuracy and visualize for only one sample"
""""""""""""
main_data_buffer = next(main_list_iterator)
print(main_data_buffer[0])
validate_data_input = per_task_dataset_loader(main_data_buffer)
#BATCH_SIZE = 64
validate_image_input = cnn_image_loader(main_data_buffer)
validate_iterator = data.DataLoader(validate_data_input, batch_size=BATCH_SIZE, drop_last=True)
image_input = validate_image_input.to(device)
images, labels, pixel, real_pixel,epoch_acc = get_predictions(model, validate_iterator,image_input, device)
EPOCHS = len(all_sample_img)//num_testing_sample
validate_acc_epoch=[]



plt.imshow(pixel[0*h*w:1*h*w].reshape(h,w), cmap='gray')
plt.title("Predict Radiography Image")
plt.show()
plt.axis('off')

pixel_real_reshape=real_pixel[0*h*w:1*h*w].reshape(h,w)
plt.imshow(pixel_real_reshape[:,:], cmap='gray')
plt.title("Ground truth Radiography Image")
plt.show()
plt.axis('off')

"plot and save tiff 3d images"
labels_pred=torch.reshape(labels[:w*w*h],(w*h,w))
all_location_loader=voxel_location

tiffpredict=torch.zeros(w, w, h, dtype=torch.long)
for i in range(w*h):
  voxel_x=np.array((all_location_loader[i][:,0]).long())[0]
  voxel_y=np.arange(0,w)
  voxel_z=np.array((all_location_loader[i][:,2]).long())[0]
  tiffpredict[voxel_x,voxel_y,voxel_z]=labels_pred[i,:].long()

plt.imshow(tiffpredict[:, :, 50], cmap='gray')
# plt.title("Slice of Predict Tomography Image")
plt.show()
plt.axis('off')


tiff_pred=np.array(tiffpredict)
tiff_pred = tiff_pred.astype('int16')
tiff_pred_0 = tiff_pred.copy()
tiff_pred_0 = np.transpose(tiff_pred, (2, 0, 1)) # reshape to image for horizontal slice in imageJ
tifffile.imsave('Artificial_sample_six_view_reconstruction.tif',tiff_pred_0)



"validate on a set of testing samples and plot accuracy and save"
"uncomment the below lines if want to validate on a set of testing samples"
""""""""""""
# for epoch in range(EPOCHS):
#     validate_acc=0
#     for i in range(num_testing_sample):
#         main_data_buffer = next(main_list_iterator)
#         #print(main_data_buffer[0])
#         validate_data_input = per_task_dataset_loader(main_data_buffer)
#         validate_image_input = cnn_image_loader(main_data_buffer)
#         validate_iterator = data.DataLoader(validate_data_input, batch_size=BATCH_SIZE, drop_last=True)
#         image_input = validate_image_input.to(device)
#         images, labels, pixel, real_pixel, task_acc = get_predictions(model, validate_iterator, image_input, device)
#         print(task_acc)
#         validate_acc += task_acc
#     validate_acc_epoch.append(validate_acc / num_testing_sample * 100)
#
#     for i in range(num_testing_sample):
#         main_data_buffer = next(main_list_iterator)
# #save
# df = pd.DataFrame({"validate_train_acc" : validate_acc_epoch})
# df.to_csv("validate_"+str(num_testing_sample)+'_samples'+".csv", index=False)
#
# # plot
# plt.figure(figsize=(10, 5))
# plt.title("Validate Accuracy on Testing Tasks")
# plt.plot(validate_acc_epoch)
# plt.xlabel("iterations")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
