"""main - Localisation from Detections - Main demo file.

This is the script to be used for running a demo of the project.

IIT - Italian Institute of Technology.
Pattern Analysis and Computer Vision (PAVIS) research line.

If you use this project for your research, please cite:
@inproceedings{rubino2017pami,
title={3D Object Localisation from Multi-view Image Detections},
author={Rubino, Cosimo and Crocco, Marco and Del Bue, Alessio},
booktitle={Pattern Analysis and Machine Intelligence (TPAMI), 2017 IEEE Transactions on},
year={2017},
organization={IEEE}

Ported to Python by Matteo Taiana.
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt

from lfd_dev import compute_estimates
from plotting import plot_est_and_gt_ellipses_on_images, plot_3D_scene

import time

####################
# 0. Introduction. #
####################
# Conventions: variable names such as Ms_t indicate that there are multiple M matrices (Ms)
# which are transposed (_t) respect to the canonical orientation of such data.
# Prefixes specify if one variable refers to input data, estimates etc.: inputCs, estCs, etc.
#
# Variable names:
# C - Ellipse in dual form [3x3].
# Q - Quadric/Ellipsoid in dual form [4x4], in the World reference frame.
# K - Camera intrinsics [3x3].
# M - Pose matrix: transforms points from the World reference frame to the Camera reference frame [3x4].
# P - Projection matrix = K * M [3*4].
#
# A note on the visibility information: when false, it might mean that either the object is not visible in the image,
# or that the detector failed. For these cases the algorithm does not visualise the estimated, nor the GT ellipses.
#
# If one object is not detected in at least 3 frames, it is ignored. The values of the corresponding ellipsoid and
# ellipses are set to NaN, so the object is never visualised.


###########################################
# 1. Set the parameters for the algorithm #
#    and load the input data.             #
###########################################
# Select the dataset to be used.
# The name of the dataset defines the names of input and output directories.
dataset = 'Aldoma'  # Data used in the original (Matlab) implementation of Lfd, published with this paper:
                    # A. Aldoma, T. Faulhammer, and M. Vincze, “Automation of ground truth annotation for multi-view
                    # rgb-d object instance recognition datasets,” in IROS 2014.

# Select whether to save output images to files.
save_output_images = True

# Load the input data.
# Data association is implicitly defined in the data structures: each column of "visibility" corresponds to one object.
# The information in "bbs" is structured in a similar way, with four columns for each object.
# bbs = np.load('Data/{:s}/InputData/bounding_boxes.npy'.format(dataset))  # Bounding boxes [X0, Y0, X1, Y1],
#                                                                          # [X0, Y0] = top-left corner
#                                                                          # [X1, Y1] = bottom-right corner
#                                                                          # [n_frames x n_objects * 4].
K = np.load('Data/{:s}/InputData/intrinsics.npy'.format(dataset))  # Camera intrinsics [3x3].
Ms_t = np.load('Data/{:s}/InputData/camera_poses.npy'.format(dataset))  # Camera pose matrices, transposed and stacked.
                                                                        # [n_frames * 4 x 3].
                                                                        # Each matrix [3x4] transforms points from the
                                                                        # World reference frame to the Camera reference
                                                                        # frame.
# visibility = np.load('Data/{:s}/InputData/visibility.npy'.format(dataset))  # Visibility information, indicates whether
#                                                                             # a detection is available for a given
#                                                                             # object, on a given frame.
#                                                                             # [n_frames x n_objects].


'''
Unknown bounding box was set to [1,1,1,2] ([X0, Y0, X1, Y1])

Example index#0:
 - test re-order of bounding box of the object
 - added unknown bounding box

original:
[
    368.97869873, 241.3981781 , 429.3835144 , 309.05929565,
    441.93484497, 259.12072754, 514.89862061, 325.12350464,
    337.46533203, 309.51794434, 386.55029297, 395.83270264,
    394.25811768, 323.92715454, 480.84927368, 399.02444458,
    326.37182617, 202.48471069, 356.46304321, 264.19009399,
    288.29751587, 249.58882141, 322.54174805, 316.91818237
]

the expected result should show only object index# 0 and 3 on frame#1

Result:
 - found wrong predict, we got #0 and #1 on frame#1 when re-order like this :
   [
        368.97869873, 241.3981781 , 429.3835144 , 309.05929565,
        394.25811768, 323.92715454, 480.84927368, 399.02444458,
        1,1,1,2,
        1,1,1,2,
        1,1,1,2,
        1,1,1,2
   ]
   So the order effect to the result in specific frame 
   but final result is OK (if the other frame is correct).
'''
bbs = np.array([[
    368.97869873, 241.3981781 , 429.3835144 , 309.05929565,
    1,1,1,2,
    1,1,1,2,
    394.25811768, 323.92715454, 480.84927368, 399.02444458,
    1,1,1,2,
    1,1,1,2],
    [268.71255493, 264.70092773, 338.02108765, 337.12896729,
    345.15518188, 286.14849854, 421.89260864, 356.74407959,
    210.70513916, 339.22924805, 276.33981323, 433.44143677,
    275.58880615, 357.95248413, 366.25387573, 438.51266479,
    233.23313904, 222.92678833, 268.73962402, 286.59417725,
    179.23153687, 274.41543579, 220.40000916, 343.59829712],
    [319.09350586, 278.32012939, 395.9881897 , 339.9385376 ,
    379.24969482, 329.80615234, 466.75039673, 386.69143677,
    217.56640625, 321.81710815, 299.9659729 , 406.78347778,
    273.675354  , 374.83673096, 360.96035767, 434.42593384,
    312.9446106 , 216.79870605, 344.60357666, 280.54574585,
    239.91426086, 248.05413818, 275.54110718, 314.31283569],
    [315.36114502, 316.1701355 , 394.37454224, 368.28683472,
    347.30505371, 378.78378296, 440.44546509, 453.36904907,
    188.36791992, 331.16345215, 276.43658447, 404.6635437 ,
    214.55780029, 385.59359741, 310.82644653, 468.81610107,
    335.87738037, 238.62005615, 369.57720947, 304.73086548,
    250.55516052, 252.22273254, 286.06552124, 319.73846436],
    [256.47576904, 320.72235107, 340.13830566, 386.76333618,
    252.72369385, 390.62054443, 347.45266724, 484.88293457,
    133.39767456, 320.96795654, 218.39674377, 377.56646729,
    132.30621338, 366.24981689, 219.84518433, 459.68603516,
    309.29263306, 250.46266174, 343.53515625, 317.23831177,
    222.79190063, 246.48374939, 258.72210693, 313.45471191],
    [347.63970947, 286.10244751, 421.15863037, 366.22714233,
    305.4281311 , 339.4513855 , 369.14886475, 436.16751099,
    259.60656738, 244.6635437 , 330.96289062, 305.32406616,
    241.35693359, 277.95428467, 285.89572144, 357.82327271,
    424.29995728, 245.79014587, 465.02682495, 313.96707153,
    358.36880493, 212.25823975, 391.81015015, 275.21115112],
    [286.1980896 , 289.27526855, 366.60769653, 373.68130493,
    246.02645874, 357.13134766, 322.96966553, 464.18991089,
    182.73692322, 254.81884766, 265.10748291, 316.26983643,
    165.15997314, 295.33834839, 224.18908691, 387.36245728,
    364.89892578, 236.72938538, 404.25271606, 307.05389404,
    289.40112305, 206.87263489, 324.01483154, 273.34524536],
    [313.42510986, 289.61465454, 398.50708008, 364.82751465,
    298.51138306, 360.96524048, 390.03161621, 462.4954834 ,
    195.63148499, 277.24972534, 278.10684204, 329.93252563,
    189.6361084 , 318.38220215, 266.52435303, 412.45333862,
    374.65100098, 225.24407959, 413.30654907, 293.85787964,
    291.13165283, 210.01704407, 325.08905029, 276.91616821]])

NUM_BB_CORNER = 4 # [X0, Y0, X1, Y1]
num_frame = int(len(Ms_t)/NUM_BB_CORNER)

current_index = 0
fulfill_indexes = {}
visibility = []

'''
Mark invisible object
'''
for bb in bbs:
    num_obj = int(len(bb) / NUM_BB_CORNER)

    visible_objs = []
    for i in range(num_obj):
        if np.array_equal(bb[i*NUM_BB_CORNER:i*NUM_BB_CORNER+NUM_BB_CORNER], [1,1,1,2]):
            visible_objs.append(False)
        else:
            visible_objs.append(True)

    visibility.append(visible_objs)

visibility = np.array(visibility)
print('Visibility:', visibility)
# visibility = np.array([
#     [ True,  True,  True,  True,  True,  True],
#     [ True,  True,  True,  True,  True,  True],
#     [ True,  True,  True,  True,  True,  True],
#     [ True,  True,  True,  True,  True,  True],
#     [ True,  True,  True,  True,  True,  True],
#     [ True,  True,  True,  True,  True,  True],
#     [ True,  True,  True,  True,  True,  True],
#     [ True,  True,  True,  True,  True,  True]])

# # Compute the number of frames and the number of objects for the current dataset from the size of the visibility matrix.
# n_frames = visibility.shape[0]
# n_objects = visibility.shape[1]


######################################
# 2. Run the algorithm: estimate the #
#    object ellipsoids.              #
######################################
stime = time.perf_counter()
[inputCs, estCs, estQs] = compute_estimates(bbs, K, Ms_t, visibility)
etime = time.perf_counter()
print(f'Compute time: {etime-stime} seconds\n')

#############################
# 3. Visualise the results. #
#############################
# Load the input images.
with open('Data/{:s}/InputData/images.bin'.format(dataset), 'rb') as handle:
    images = pickle.load(handle)

# Load Ground Truth ellipsoids.
gtQs = np.load('Data/{:s}/GroundTruthData/gt.npy'.format(dataset), allow_pickle=True)
# gtQs = np.zeros(shape=(0))

# Plot ellipses on input images.
plot_est_and_gt_ellipses_on_images(K, Ms_t, estCs, gtQs, visibility, images, dataset, save_output_images)

# Plot ellipsoids and camera poses in 3D.
plot_3D_scene(estQs, gtQs, Ms_t, dataset, save_output_images)

# Visualise the plots that have been produced.
plt.show()
