"Import packages"
import random
import numpy as np
import torch
import learn2learn as l2l
from torch import nn, optim
import os
import pandas as pd

from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import cv2
import os
import glob
import zipfile
import functools
from tqdm import tqdm

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

"Main Parameters"
acc_epoch=0#1810

# Number of training and testing dataset
# num_training_sample = 1800
# num_testing_sample= 32
num_training_sample = 56
num_testing_sample= 14
#image size
h = 100
w = 150

# batch size
BATCH_SIZE = 4096 #for number of probing X-rays per samples
IMAGE_BATCH_SIZE=36 #for number of samples

weight_factor = 25  # weight factor for MSE for 2D labels

"Load and order data path"
# Data path for training and testing datasets
DATA_PATH = "Examples_of_training_and_testing_datasets"
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
files = glob.glob(os.path.join(DATA_PATH, '*.tiff'))
files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

for infile in files:
    all_tiff.append(infile)
    all_tiff = all_tiff

# each task: 1 3D tiff and 12 2D png data path
sample_img_path = []
png_per_sample = []
tiff_per_sample = []
img_per_sample = []
all_sample_img = []

"remove the samples if it is out of cutoff range"
h_cutoff_down = 100
h_cutoff_up = 250
w_cutoff_down = 150
w_cutoff_up = 154

"Load image file path to a list"
for i in range(len(all_tiff)-acc_epoch):  # len(all_tiff)
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


"""Input Parameters"""

"parameters for rotation"
# rotate the x-ray beam (rotate voxel location)
# tiff img rotate counter clock-wise
# is equivalent to
# x-ray beam location rotate clock-wise
#rotate_angle_counter_clock = np.arange(15, 180, 15)# 15 deg
"Adjust for different number of view reconstruction"
rotate_angle_counter_clock = np.arange(30, 180, 30) #six-view reconstruction
# rotate_angle_counter_clock = np.arange(45, 180, 45) #four-view reconstruction
# rotate_angle_counter_clock = 90*np.ones(1) #two-view reconstruction

rotate_angle_clock = -rotate_angle_counter_clock  # because artificial radiography is created by rotating tiff counterclockwise, and make projection. So the x-ray beam is considered rotate clockwise
center = (w / 2, w / 2)
n_projection = len(rotate_angle_counter_clock) + 1

"parameters for radiography imaging"

# Imaging properties
actual_resolution_um = 26.33  # um/pixel
mu_l_p = 1/4064.03  # linear attenuation coefficient of particle [1/um] @30keV
mu_l_quartz = 1/5835.76  # linear attenuation coefficient of quartz [1/um]
eff_detector = 0.45  # quantum efficiency


# Determine quartz wall thickness with x
quartz_thickness = np.zeros(w)
OR = 5.95 / 2  # outside radius [mm]
IR = 3.95 / 2  # inside radius [mm]
x_quartz = np.linspace(-IR, IR, w, endpoint=True)
for i in range(w):
    quartz_thickness[i] = 1000 * 2 * (
            np.sqrt([OR ** 2 - x_quartz[i] ** 2]) - np.sqrt([IR ** 2 - x_quartz[i] ** 2]))  # [um]

quartz_thickness = torch.from_numpy(quartz_thickness)


"Properties for adjust pixel grayscale (strech histogram to [0,1])"
# avg and sigma are estimated by averaging the original figures of 3 samples in each type
avg_origin =0.20782
sigma_origin = 0.02994
n_std = 3.5
min_grayscale_bound = avg_origin - n_std * sigma_origin
max_grayscale_bound = avg_origin + n_std * sigma_origin  # average and sigma of original photo (before adjusted are found in matlab e.g. average the value for several test)

"Functions"
def crop_img(img, cropx, cropy):
    # define a img function to horizontally center crop and vertically crop from one edge
    startx = (img.shape[1] - cropx) // 2
    endx = startx + cropx
    endy = cropy
    return img[0:endy, startx:endx]


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



def cnn_image_loader(iterator):
    fixed_projection = io.imread(iterator[0])
    print(iterator[0])
    fixed_projection = np.flip(fixed_projection,
                               0)
    fixed_projection = crop_img(fixed_projection, w, h)/ 255

    #add 90deg
    fixed_projection_90= io.imread(iterator[6])
    fixed_projection_90 = np.flip(fixed_projection_90,
                               0)
    fixed_projection_90 = crop_img(fixed_projection_90, w, h)/ 255

    fixed_projection = np.concatenate((fixed_projection, fixed_projection_90), axis=0)

    mean,std=fixed_projection.mean(),fixed_projection.std()
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), transforms.Resize(size=[224,224])])
    fixed_projection = preprocess(fixed_projection)
    fixed_projection = fixed_projection.type(torch.float32)
    fixed_projection = torch.stack([fixed_projection], dim=0)
    return fixed_projection


"Functions for training"
def calculate_accuracy(y_pred, y):
    y_pred = y_pred.detach().clone()
    y_pred = y_pred.argmax(1, keepdim=True)
    correct = y_pred.eq(y).sum()
    acc = correct.float() / y.shape[0]
    return acc

def reconstruc_radiography(y_pred, original_projection_location_x):  # add x[0]
    y_pred = y_pred[:, 1].float()
    y_pred = torch.reshape(y_pred, (-1, w))
    Grayscale_beam = eff_detector * torch.exp(
        -mu_l_p * actual_resolution_um * torch.sum(y_pred, dim=1) - mu_l_quartz * quartz_thickness[
            original_projection_location_x.long()])
    Grayscale_beam_adjusted = (Grayscale_beam - min_grayscale_bound) * (1 - 0) / (
            max_grayscale_bound - min_grayscale_bound) + 0
    return Grayscale_beam_adjusted


def calculate_accuracy(y_pred, y):
    y_pred = y_pred.detach().clone()
    y_pred = y_pred.argmax(1, keepdim=True)
    correct = y_pred.eq(y).sum()
    acc = correct.float() / y.shape[0]
    return acc


def reconstruc_radiography(y_pred, original_projection_location_x):  # add x[0]
    y_pred = y_pred[:, 1].float()
    y_pred = torch.reshape(y_pred, (-1, w))
    Grayscale_beam = eff_detector * torch.exp(
        -mu_l_p * actual_resolution_um * torch.sum(y_pred, dim=1) - mu_l_quartz * quartz_thickness[
            original_projection_location_x.long()])
    Grayscale_beam_adjusted = (Grayscale_beam - min_grayscale_bound) * (1 - 0) / (
            max_grayscale_bound - min_grayscale_bound) + 0
    return Grayscale_beam_adjusted

"prepare voxel_location for x-ray beams -> original location without rotation"
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
x_location = torch.tensor(x_location, dtype=torch.int32)

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


voxel_input_origin = []
location_buffer_origin = []
location_buffer_origin = torch.cat(voxel_rotate_location, 0)
voxel_input_origin = torch.cat((voxel_location, location_buffer_origin), axis=0)

voxel_label=torch.arange(len(voxel_input_origin)).long()

"uncomment and adjust n_positional_encoding if want to add positional encoding"
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

"Dataloader preparation"

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

    # load radiography label change the 2*j
    all_projection_png = []
    for j in range(n_projection):
        "adjust below if want to change the number of views for reconstruction"
        projection_img = io.imread(img_list_iterator[2*j]) / 255  # six-view reconstruction
        # projection_img = io.imread(img_list_iterator[3*j]) / 255  # four-view reconstruction
        # projection_img = io.imread(img_list_iterator[6*j]) / 255  # two-view reconstruction

        #projection_img = io.imread(img_list_iterator[j]) / 255  # find the first figure
        projection_img = np.flip(projection_img,
                                 0)  # flip verticle #same accuracy, flip can have a lower loss, so this could be correct
        projection_img = crop_img(projection_img, w, h)
        projection_img = projection_img.flatten()
        if j == 0:
            all_projection_png = projection_img
        else:
            all_projection_png = np.concatenate((all_projection_png, projection_img), axis=0)

        all_radiograph = list(torch.tensor(all_projection_png.copy(), dtype=torch.float32))

    per_task_dataset = [voxel_label, occupancy_label, all_radiograph]  # voxel location of x-ray beam imput, tomography 3d label
    per_task_dataset = [[voxel_input_query, three_d_label_query, two_d_label_query] for (voxel_input_query, three_d_label_query, two_d_label_query) in
                     zip(per_task_dataset[0], per_task_dataset[1], per_task_dataset[2])]

    return per_task_dataset

d1 = torch.linspace(-1, 1, w)
d2=torch.linspace(-1,1,h)
meshx, meshy,meshz = torch.meshgrid((d1, d1,d2))

"remove last layer of resnet"
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
        # self.layer0 = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
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
        #self.fc = torch.nn.Linear(filters[4], 2048)
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

#resnet18_model = ResNet(1, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=512) #change this if want a resnet18
resnet18_model = ResNet(1, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=512) # this is resnet34

class Net_upsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=resnet18_model
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
        y=F.relu(self.bn0(y))
        y = F.relu(self.bn1(self.convt3d_1(y)))
        y = F.relu(self.bn_c_1(self.conv_1(y)))
        y = F.relu(self.bn2(self.convt3d_2(y)))
        y = F.relu(self.bn_c_2(self.conv_2(y)))
        y = F.relu(self.bn3(self.convt3d_3(y)))
        y = F.relu(self.bn_c_3(self.conv_3(y)))


        y = F.sigmoid(self.convt3d_4(y))
        if torch.Tensor.dim(x)==3:
            x=x.unsqueeze(0)
            grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
            y = F.grid_sample(y, grid.to(device), mode='bilinear', padding_mode='zeros').squeeze(0)
        else:
            grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
            y = F.grid_sample(y, grid.to(device), mode='bilinear', padding_mode='zeros').squeeze()

        x=x.view(x.shape[0], -1, x.shape[-1]) # reshape to [image batch size, w*voxel_batch_size, feature dim =3]
        x=x[:, :, 0:3]
        y_pred=[]
        for i in range(x.shape[0]):
            voxel_x = x[i, :, 0]
            voxel_y = x[i, :, 1]
            voxel_z = x[i, :, 2]
            y_pred_per_batch = torch.zeros(x.shape[1], dtype=torch.float32).to(device)
            idx_inside_bound = (voxel_x >= 0) & (voxel_x < w) & (voxel_y >= 0) & (voxel_y < w)
            y_pred_per_batch[idx_inside_bound] =y[i][
                voxel_x[idx_inside_bound].long(), voxel_y[idx_inside_bound].long(), voxel_z[idx_inside_bound].long()]
            y_pred.append(y_pred_per_batch)
        y_pred=torch.cat(y_pred, dim=0)
        y_pred = y_pred.view(len(y_pred), 1)
        y_pred=torch.cat((1 - y_pred, y_pred), 1) #return two
        return y_pred


resnet18=Net_upsampling().cnn
model=Net_upsampling()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#load model
print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9,weight_decay=0.000005)
criterion_1 = nn.CrossEntropyLoss()
criterion_2 = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()

"load model"
model.load_state_dict(torch.load('six_view_sphere_cnn_model_EPOCH_0.pth'))

criterion_1 = criterion_1.to(device)
criterion_2 = criterion_2.to(device)
quartz_thickness=quartz_thickness.to(device)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def iterator_set(num_training_sample):
    "return set of data and image iterators for random shuffle"
    data_iterator_set=[]
    image_iterator_set=[]
    for i in range(num_training_sample):
        main_data_buffer = next(main_list_iterator)
        print(main_data_buffer[0],i)
        data_input = per_task_dataset_loader(main_data_buffer)
        data_iterator = data.DataLoader(data_input,shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
        data_iterator_set.append(data_iterator)
        image_input = cnn_image_loader(main_data_buffer)
        image_iterator_set.append(image_input)
    return data_iterator_set, image_iterator_set

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

def batch_order_set_meta(num_of_train,iterator_set,image_batch_size): #tps is task per step
    #overall_iterator_len=len(iterator_set['input_data_0'])*num_of_train
    single_list=np.arange(num_of_train)
    all_list=np.repeat(single_list, len(iterator_set[0]))
    all_list = torch.tensor(all_list, dtype=torch.float32).long()
    batch_order_iterator=data.DataLoader(all_list,shuffle=True, batch_size=image_batch_size,drop_last=True)
    # if use the same order
    return batch_order_iterator # ust batch_order.next() to get the shuffle list

def batch_order_set_testing(num_of_test,iterator_set,image_batch_size): #tps is task per step
    #overall_iterator_len=len(iterator_set['input_data_0'])*num_of_train
    single_list=np.arange(num_of_test)
    all_list=np.repeat(single_list, len(iterator_set[0]))
    all_list = torch.tensor(all_list, dtype=torch.float32).long()
    batch_order_iterator=data.DataLoader(all_list, batch_size=image_batch_size,drop_last=True)
    # if use the same order
    return batch_order_iterator



def batch_data_preparation(iterator_set,image_iterator_set, image_batch_size,batch_order):
    "shuffle the order"
    image_batch=image_iterator_set[batch_order[0]]
    x_set,y_set,z_set=iterator_set[f'input_data_{batch_order[0]}'].next()
    x_set,y_set=x_set.unsqueeze(0),y_set.unsqueeze(0)
    for i in range(image_batch_size-1):
        x, y, z = iterator_set[f'input_data_{batch_order[i+1]}'].next()
        image_batch=torch.cat((image_batch,image_iterator_set[batch_order[i+1]]),dim=0)
        x_set,y_set,z_set=torch.cat((x_set,x.unsqueeze(0)),dim=0),torch.cat((y_set,y.unsqueeze(0)),dim=0),torch.cat((z_set,z),dim=0)
    return x_set, y_set, z_set, image_batch



def train_meta(model, iterator_set,image_set,batch_order_list, optimizer, criterion_1, criterion_2, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for i in range(len(batch_order_list)):
        label,y,z,batch_image = batch_data_preparation(iterator_set, image_set,IMAGE_BATCH_SIZE,
                                             batch_order_list.next())
        "input location"
        x = voxel_input_origin[label].to(device)
        "ground truth occupancy"
        y = y.to(device).reshape(-1, 1).long()
        "ground truth greyvalue"
        z = z.to(device)
        "original x location for finding quartz"
        t = x_location[label].to(device).view(-1)
        "batch image input"
        batch_image=batch_image.to(device)
        y_pred = model(x, batch_image)
        grayscale_pred = reconstruc_radiography(y_pred, t).float()
        acc = calculate_accuracy(y_pred, y)
        loss_1 = criterion_1(y_pred, y.squeeze())
        loss_2 = criterion_2(grayscale_pred, z)
        loss =loss_1+weight_factor* loss_2

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i%(len(iterator_set['input_data_0'])//num_training_sample)==0:
        #     print(acc_step/len(input_data_holder['input_data_0']))
        #     acc_step = 0
        #     torch.save(model.state_dict(),
        #                'F:\\particle_data\\Trained_model\\train_cnn\\data_output_3\\cross_task_model_EPOCH_backup.pth')
    return epoch_loss / len(batch_order_list), epoch_acc / len(batch_order_list)



def evaluate_meta(model, iterator_set,image_set,batch_order_list, criterion_1, criterion_2, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    #for (x, y, z, t) in tqdm(iterator, desc="Training", leave=False):
    with torch.no_grad():
        for i in range(len(batch_order_list)):
            label,y,z,batch_image = batch_data_preparation(iterator_set, image_set,IMAGE_BATCH_SIZE,
                                                 batch_order_list.next())
            "input location"
            x = voxel_input_origin[label].to(device)
            "ground truth occupancy"
            y = y.to(device).reshape(-1, 1).long()
            "ground truth greyvalue"
            z = z.to(device)
            "original x location for finding quartz"
            t = x_location[label].to(device).view(-1)
            "batch image input"
            batch_image=batch_image.to(device)
            y_pred = model(x, batch_image)
            grayscale_pred = reconstruc_radiography(y_pred, t).float()
            acc = calculate_accuracy(y_pred, y)

            loss_1 = criterion_1(y_pred, y.squeeze())
            loss_2 = criterion_2(grayscale_pred, z)
            loss =loss_1+weight_factor* loss_2
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(batch_order_list), epoch_acc / len(batch_order_list)

"prepare training and testing set"
per_training_iterator_set_train, per_training_image_iterator_set_train = iterator_set(num_training_sample)
per_testing_iterator_set_test, per_testing_image_iterator_set_test= testing_iterator_set(num_testing_sample)

EPOCHS = 20
#EPOCHS=1
test_losses = []
train_losses = []
test_accs = []
train_accs = []
best_valid_loss = float('inf')



"#######################"
for epoch in range(EPOCHS):
    "Training data"
    batch_order_origin = batch_order_set_meta(num_training_sample, per_training_iterator_set_train, IMAGE_BATCH_SIZE)

    per_training_iterator_set=per_training_iterator_set_train.copy()
    per_training_image_iterator_set=per_training_image_iterator_set_train.copy()
    #batch_order_iterator = copy.deepcopy(batch_order_origin)
    batch_order = iter(batch_order_origin)
    input_data_holder = {}
    input_image_holder = {}
    for i in range(num_training_sample):
        input_data_holder['input_data_' + str(i)] = iter(per_training_iterator_set[i])
        #input_image_holder['input_image_' + str(i)] = per_training_image_iterator_set[i]
    locals().update(input_data_holder)

    "Training"

    start_time = time.monotonic()
    #train_loss, train_acc = train_set(model, input_data_holder, input_image_holder,batch_order, optimizer, criterion_1, criterion_2, device)
    train_loss, train_acc = train_meta(model, input_data_holder, per_training_image_iterator_set, batch_order, optimizer, criterion_1,
                                      criterion_2, device)
    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch :02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    train_losses.append(train_loss)
    train_accs.append(train_acc * 100)

    "Testing data"
    batch_order_testing = batch_order_set_testing(num_testing_sample, per_testing_iterator_set_test, IMAGE_BATCH_SIZE)

    per_testing_iterator_set=per_testing_iterator_set_test.copy()
    per_testing_image_iterator_set=per_testing_image_iterator_set_test.copy()
    #batch_order_iterator_testing = copy.deepcopy(batch_order_testing)
    batch_order_test = iter(batch_order_testing)
    testing_input_data_holder = {}
    for i in range(num_testing_sample):
        testing_input_data_holder['input_data_' + str(i)] = iter(per_testing_iterator_set[i])
        #input_image_holder['input_image_' + str(i)] = per_training_image_iterator_set[i]
    locals().update(testing_input_data_holder)

    start_time = time.monotonic()
    test_loss, test_acc = evaluate_meta(model, testing_input_data_holder,per_testing_image_iterator_set, batch_order_test,criterion_1, criterion_2, device)
    end_time = time.monotonic()
    #scheduler.step(test_loss)
    # if test_loss < best_valid_loss:
    #     best_valid_loss = test_loss
    #     #torch.save(model.state_dict(), 'tut-model.pt')
    #     torch.save(model.state_dict(),
    #                'F:\\particle_data\\Trained_model\\train_cnn\\data_output_3\\try.pth')
    #torch.save(model.state_dict(),'F:\\particle_data\\Trained_model\\train_cnn\\data_output_3\\six_view_sphere_cnn_model_EPOCH_' + str(epoch) + '.pth')
    torch.save(model.state_dict(),'six_view_EPOCH_' + str(epoch) + '.pth')

    print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch :02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    # test_losses_epoch += test_loss
    # test_accs_epoch += test_acc
    test_losses.append(test_loss)
    test_accs.append(test_acc * 100)

print(test_loss)

"##########################################################################"
df = pd.DataFrame({"train loss" : train_losses, "train acc": train_accs, "test loss": test_losses, "test acc": test_accs})

plt.figure(figsize=(10, 5))
plt.title("Training and Testing Loss")
plt.plot(test_losses, label="test")
plt.plot(train_losses, label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Training and Testing Accuracy")
plt.plot(test_accs, label="test")
plt.plot(train_accs, label="train")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()