# -*- coding: utf-8 -*-
"""
# Product Recognition on Store Shelves

- Fabian Vincenzi

### Import the libraries
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

"""## Step A - Multiple Product Detection:

- Declare the SIFT and the dirname
- Load scene image
- Load product image
"""

sift = cv2.xfeatures2d.SIFT_create()

#set the global directory
dirname = 'object_detection_project/'

#read the scene image

#set the dir for test img and their name, prepare the dictionary with images,
#keypoints and descriptor
dirname_scene = 'scenes/'
imgs_scene = ['1', '2', '3', '4', '5']
img_scene = {}

#save every images, keypoints and descriptor
for j in imgs_scene:

    img = cv2.imread(dirname + dirname_scene + 'e' + j + '.png')
    kp_scene = sift.detect(img)
    kp_scene, des_scene = sift.compute(img, kp_scene)
    img_scene[j] = [img, kp_scene, des_scene]

#read the product image

#set the dir for product img and their name, dictionary with images, keypoints
#and descriptor
dirname_prod = 'models/'
imgs_prod = ['0', '1', '11', '19', '24', '25', '26']
img_prod = {}

INDEX_IMAGE = 0
INDEX_KP = 1
INDEX_DES = 2

#save every image, keypoints and descriptor in the dictionary
for i in imgs_prod:

    img = cv2.imread(dirname + dirname_prod + i + '.jpg')
    kp_prod = sift.detect(img)
    kp_prod, des_prod = sift.compute(img, kp_prod)
    img_prod[i] = [img, kp_prod, des_prod]

"""Define a function that find matches between keypoints of the product and keypoints of the scene, and if they exceed a threshold collect them in a list"""

#function to find the product in the shelves
def object_retrieve(prod_des, scene_des, threshold = 0.45, k=2):


    FLANN_INDEX_KDTREE = 1

    # define algorithm param
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # inizialize matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching
    matches = flann.knnMatch(prod_des, scene_des, k)

    # array that will contain all the 'good' matches appllying a threshold
    good = []
    for m,n in matches:
        if m.distance < threshold *n.distance:
            good.append(m)

    return good

"""Function needed in the code"""

#function to calculate the distance of 2 points
def distance_2points(A, B):
    return math.sqrt(np.power(A[0] - B[0], 2) + np.power(A[1] - B[1], 2))

#function to solve the exceeding dimension
def correct_box_exceeding(corners, scene_img):

    corners_no_exceeding = []

    #adjust overdimension
    scene_height = scene_img.shape[0]
    scene_width = scene_img.shape[1]

    for index, [corner_top_left, corner_bot_right] in enumerate(corners):

        corner_top_left = list(corner_top_left)
        corner_bot_right = list(corner_bot_right)

        #corners that goes out on the left of the image
        if corner_top_left[0] < 0:
            corner_top_left[0] = 0

        if corner_bot_right[0] < 0:
            corner_bot_right[0] = 0

        #corners that goes out on the right of the image
        if corner_top_left[0] > scene_width:
            corner_top_left[0] = scene_width

        if corner_bot_right[0] > scene_width:
            corner_bot_right[0] = scene_width

        #corners that goes out on the top of the image
        if corner_top_left[1] < 0:
            corner_top_left[1] = 0

        if corner_bot_right[1] < 0:
            corner_bot_right[1] = 0

        #corners that goes out on the bot of the image
        if corner_top_left[1] > scene_height:
            corner_top_left[1] = scene_height

        if corner_bot_right[1] > scene_height:
            corner_bot_right[1] = scene_height



        corner_top_left = tuple(corner_top_left)
        corner_bot_right = tuple(corner_bot_right)

        corners_no_exceeding.append([ (int(corner_top_left[0]),
                                      int(corner_top_left[1])),
                                      (int(corner_bot_right[0]),
                                      int(corner_bot_right[1]))])

    return corners_no_exceeding

# function to solve the color problem in the image
def correct_color_problem(corners, scene, i, j, N=3, M=4,
                         DIFF_COLOR_CHANNEL = 50, MAX_NO_GOOD_CELLS = 2):

    corners_solution = []

    for index, [box_top_left, box_bot_right] in enumerate(corners):

        prod_bins = image_in_bins(img_prod[i][INDEX_IMAGE], N, M)

        scene_bins = image_in_bins(scene[j][INDEX_IMAGE][box_top_left[1]:box_bot_right[1],
                                              box_top_left[0]:box_bot_right[0]],
                                  N, M)


        if not prod_bins or not scene_bins:
            return []

        good = True
        No_good = 0


        #cycle the prod dictionary, get the key of the scene and the diff of the means
        for k, v in prod_bins.items():

            means_scene_k = scene_bins.get(k)

            # bounding box correct if the difference of the bins in the 3 channels,
            # are below a certain threshold
            r_diff = np.absolute(v[0] - means_scene_k[0])
            g_diff = np.absolute(v[1] - means_scene_k[1])
            b_diff = np.absolute(v[2] - means_scene_k[2])

            if r_diff >= DIFF_COLOR_CHANNEL or g_diff >= DIFF_COLOR_CHANNEL or b_diff >= DIFF_COLOR_CHANNEL:

                No_good += 1


        if No_good < MAX_NO_GOOD_CELLS:
            corners_solution.append([ (int(box_top_left[0]), int(box_top_left[1])),
                                    (int(box_bot_right[0]), int(box_bot_right[1]))])


    return corners_solution

#function that split an image in bins and return a dictionary with the
# means of the 3 channels
def image_in_bins(img, N=3, M=4):

    img_bins = {}

    img_height = img.shape[0]
    img_width = img.shape[1]

    height_step = int(img_height / M)
    width_step = int(img_width / N)

    r = 0

    if img_height != 0 and height_step != 0 and img_width != 0 and width_step != 0:

        for row in np.arange(0, img_height, height_step):
            c = 0

            for col in np.arange(0, img_width, width_step):

                if row + 2 * height_step > img_height and col + 2 * width_step > img_width:

                    channel_r, channel_g, channel_b = cv2.split(img[row:, col:])

                elif row + 2 * height_step > img_height:

                    channel_r, channel_g, channel_b = cv2.split(img[row: row + height_step, col:])

                else:

                    channel_r, channel_g, channel_b = cv2.split(img[row: row + height_step,
                                                                   col : col + width_step])

                if r < M  and c < N:

                    #save the means of the 3 channels
                    img_bins[r, c] = (np.mean(channel_r), np.mean(channel_g), np.mean(channel_b))


                c += 1

            r += 1


        return img_bins

# funciton that plot the final corners
def plot_box_dark_area(img, corners):

    if len(corners) > 0:

        for box_top_left, box_bot_right in corners:

            img_box = cv2.rectangle(img, box_top_left, box_bot_right, (0,0,0), -1)

        return img_box

    else:

        return None

#function that plot the corner in the shelves
def plot_box_corners(img, corners, thickness=10):

    if len(corners) > 0:

        for box_top_left, box_bot_right in corners:
            scene_img_with_box = cv2.rectangle(img, box_top_left, box_bot_right, (0, 255, 0), thickness)

        return scene_img_with_box

    else:

        return None

def print_result(results, scene):

    for i in imgs_prod:
        print('Product {} - {} instance/s found:'.format(i, results[scene][i]['count']))
        n = 0

        if results[scene][i].get('width', None):
            n+=1
            print('\t Instance {} position: {}, width: {}px, height: {}px'.format(n, results[scene][i]['pos'], results[scene][i]['width'], results[scene][i]['height'] ))

    print('_' * 100 + '\n')

# results of the code that will be printed
retrieve_results = {}

scene_with_block = {}

min_match_count = 20

BINS_WIDTH = 3
BINS_HEIGHT = 4
DIFF_COLOR_CHANNEL = 79
MAX_NO_GOOD_CELLS = 1
THRESHOLD = 0.5


for j in imgs_scene:

    retrieve_results[j] = {}
    scene_with_block[j] = np.copy(img_scene[j][INDEX_IMAGE])

    for i in imgs_prod:

        retrieve_results[j][i] = {}
        retrieve_results[j][i]['count'] = 0
        scene_one_product = np.copy(img_scene[j][INDEX_IMAGE])


        finding = True

        #the model is in the image if there are enough matches
        while finding:

            kp_scene = sift.detect(scene_one_product)
            kp_scene, des_scene = sift.compute(scene_one_product, kp_scene)

            matches = object_retrieve(img_prod[i][INDEX_DES], des_scene, THRESHOLD, 2)

            if len(matches) >= min_match_count:

                src_pts = np.float32([img_prod[i][INDEX_KP][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Calculating homography based on correspondences
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Apply homography to project corners of the query image into the image
                h, w = img_prod[i][INDEX_IMAGE].shape[:2]
                pts = np.float32([ [0,0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                box_top_left = dst[0][0]
                box_bot_left = dst[1][0]
                box_bot_right = dst[2][0]
                box_top_right = dst[3][0]

                box_width = int(distance_2points(box_top_left, box_top_right))
                box_height = int(distance_2points(box_top_left, box_bot_left))

                if box_width < box_height:


                    box_corners = []
                    box_corners.append([box_top_left, box_bot_right])

                    # correct exceeding box
                    box_corners_no_exceeding = correct_box_exceeding(box_corners, img_scene[j][0])

                    # correct color problem with bins
                    box_corners_no_color_prob = correct_color_problem(
                        box_corners_no_exceeding,
                        img_scene,
                        i, j, N = BINS_WIDTH, M = BINS_HEIGHT,
                        DIFF_COLOR_CHANNEL = DIFF_COLOR_CHANNEL,
                        MAX_NO_GOOD_CELLS = MAX_NO_GOOD_CELLS)


                    # correct color problem without bins
                    if len(box_corners_no_color_prob) > 0:

                        finding = False

                        for [corn_top_left, corn_bot_right] in box_corners_no_color_prob:

                            retrieve_results[j][i]['count'] += 1
                            retrieve_results[j][i]['width'] = box_width
                            retrieve_results[j][i]['height'] = box_height
                            retrieve_results[j][i]['pos'] = (int(corn_top_left[0]), int(corn_top_left[1]))


                    else:

                        for [corn_top_left, corn_bot_right] in box_corners_no_exceeding:

                            scene_one_product = plot_box_dark_area(scene_one_product,
                                                                box_corners_no_exceeding)


                    img_box = plot_box_corners(scene_with_block[j], box_corners_no_color_prob)

                    if img_box is not None:
                        scene_with_block[j] = img_box


            else:
                finding = False

"""Final output of Step A"""

for j in imgs_scene:
    plt.imshow(cv2.cvtColor(scene_with_block[j], cv2.COLOR_BGR2RGB))
    plt.show()
    print_result(retrieve_results, j)

"""## Step B - Multiple Object Detection

We need another libreries
"""

import operator

"""Compute the center of the image. Knowing that the cereal box has a rectangular shape, the center coordinates are the half of the vertical and horizontal dimensions of the product."""

# find the product center and stores it
def FindCenter(img, i, INDEX_CENTER=3):

    h, w, channels = img[i][INDEX_IMAGE].shape
    ver = int(h/2) # vertical coordinate of the center
    hor = int(w/2) # horizontal coordinate of the center
    C = np.array([hor, ver]) #center of the model

    img_prod[i][INDEX_CENTER] = C # save the center in the product

"""Compute the vectors starting from the keypoints of the product and pointing the center C it is possible to identify the distance and the directions of each keypoint position with respect to C"""

# create or update the img_prod with a list containing the vote of the vector for each good match
# between keypoints and the product

def VoteVectors(matches, prod_img, i):

    # getting the coordinate of the center
    C_x = prod_img[i][INDEX_CENTER][0]
    C_y = prod_img[i][INDEX_CENTER][1]

    # initialize vector V
    V =[]

    for m in matches:

        #get the coordinate of m-th keypoints and compute the point KP - C
        Kp_x = int(prod_img[i][INDEX_KP][m.queryIdx].pt[0])
        kp_y = int(prod_img[i][INDEX_KP][m.queryIdx].pt[1])

        # defining the point V
        vx = Kp_x - C_x
        vy = kp_y - C_y

        # compute the slope, aligning center point C and the m-th keypoyint
        # if V=(0,0), C and kp are coincident and the slope doesn't exist
        # if they are aligned in a vertical line slope doesn't exist too
        # the array contain two information, the first is the actual value of the slope, the second if the slope
        # exist or not (1, 0 respectively)

        if (np.abs(vx) + np.abs(vy)) == 0:
            slope = [0,0]
        elif vy == 0:
            slope = [1,0]
        else:
            slope = [vx/vy, 1]

        # save the slope
        V.append([vx, vy, slope])

    # if there wasn't the vector V append it otherwise update it
    if len(prod_img[i]) <= INDEX_V:

        prod_img[i].append(V)

    else:

        prod_img[i][INDEX_V] = V


    return V

"""Building an accumulator array in order to estimate the position of the center of each instance of the product in the scene"""

# compare the kypoint and the angle and define the mean scale factor and the mena relative angle
def compare_keypoints(prod, scene, matches):

    size_prod_kp = [] #keypoints product
    size_scene_kp = [] #keypoints scene
    size_ratio = [] # ratio keypoints product and scene

    angle_prod_kp = [] # matched keypoint product
    angle_scene_kp = [] #matched keypoint scene
    angle_relatives = [] #rotation between product and scene


    for m in matches:

        # 2 variables with the size of the keypoints
        size_kp_P = np.float32(prod[INDEX_KP][m.queryIdx].size)
        size_kp_S = np.float32(scene[INDEX_KP][m.trainIdx].size)
        size_prod_kp.append(size_kp_P)
        size_scene_kp.append(size_kp_S)

        #2 variables with the angle of the keypoints
        angle_kp_P = np.float32(prod[INDEX_KP][m.queryIdx].angle)
        angle_kp_S = np.float32(scene[INDEX_KP][m.trainIdx].angle)
        angle_prod_kp.append(angle_kp_P)
        angle_scene_kp.append(angle_kp_S)

        # define change in dimension between kp of the product and the scene
        size_ratio.append(size_kp_P / size_kp_S)

        # define change in dimension between kp of the produc and scene
        angle_relatives.append(angle_kp_P - angle_kp_S)

    # compute the Mean scale factor
    if len(size_ratio):
        Mean_Scale_Factor = (sum(size_ratio)) / len(size_ratio)

    else:
        Mean_Scale_Factor = 3.5

    # compute the mean relative angle
    if len(angle_relatives):
        Mean_Relative_Angle = (sum(angle_relatives)) / len(angle_relatives)

    else:
        Mean_Relative_Angle = 0


    return Mean_Scale_Factor, Mean_Relative_Angle

# create and array with the position of the center
def Array_acc(DIMENSION1_CELL, DIMENSION2_CELL, scene, prod, matches, V, i, j):

    #set Accumulator
    Acc_dim = (int(scene[INDEX_IMAGE].shape[0] / DIMENSION2_CELL),
              int(scene[INDEX_IMAGE].shape[1] / DIMENSION1_CELL))

    Acc_points = {}

    Acc_array = np.zeros(Acc_dim)

    #get keypoints of the scene that are good matches
    pts_scene = np.float32([scene[INDEX_KP][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    #compute the mean scale factor and the mean angle between product and scene
    r, alpha = compare_keypoints(prod, scene, matches)

    # list with C estimation
    scene_C = []

    for m in range(len(matches)):

        #rescaling Vote Vector
        scene_Vx = V[m][0] / r
        scene_Vy = V[m][1] / r

        #compute estimated position of the center
        scene_Cx = pts_scene[m][0][0] - scene_Vx
        scene_Cy = pts_scene[m][0][1] - scene_Vy

        # store the estimation
        scene_C.append([scene_Cx, scene_Cy])

        C_x = int(scene_Cx / (DIMENSION1_CELL))
        C_y = int(scene_Cy / (DIMENSION2_CELL))

        if C_x in range(Acc_dim[1]):
            if C_y in range(Acc_dim[0]):

                Acc_array[C_y, C_x] += 1

                # save the scene point
                if not(C_y, C_x, 'S') in Acc_points:
                    Acc_points[(C_y, C_x, 'S')] = []
                    Acc_points[(C_y, C_x, 'S')].append((scene_Cx, scene_Cy))

                else:

                    Acc_points[(C_y, C_x, 'S')].append((scene_Cx, scene_Cy))


    return Acc_array, scene_C, Acc_points

"""Function that estimates the centers of the product in the scene"""

# estimate the center of the image
def estimate_center(prod, scene, matches, V):

    #extract the keypoints of the scene
    pts_scene = np.float32([scene[INDEX_KP][m.trainIdx].pt for m in matches]).reshape(-1,1,2)

     #compute the mean scale factor and the mean angle between product and scene
    r, alpha = compare_keypoints(prod, scene, matches)

    #list with C estimation
    C_scene = []

    for m in range(len(matches)):

        #rescaling voting vectors
        Vx_scene = V[m][0] / r
        Vy_scene = V[m][1] / r

        #compute estimate position
        Cx_scene = pts_scene[m][0][0] - Vx_scene
        Cy_scene = pts_scene[m][0][1] - Vy_scene

        C_scene.append([Cx_scene, Cy_scene])


    return C_scene, r

"""Function that solve the color problem, slightly different from the one in Step A"""

# function to solve the color problem in the image
def correct_color_problem(corners, scene, i, j, N=3, M=4,
                         DIFF_COLOR_CHANNEL = 50, MAX_NO_GOOD_CELLS = 2):

    corners_solution = []

    for index, [box_top_left, box_bot_right] in enumerate(corners):

        prod_bins = image_in_bins(img_prod[i][INDEX_IMAGE], N, M)

        scene_bins = image_in_bins(scene[j][INDEX_IMAGE][box_top_left[1]:box_bot_right[1],
                                              box_top_left[0]:box_bot_right[0]],
                                  N, M)


        if not prod_bins or not scene_bins:
            return []

        good = True


        #cycle the prod dictionary, get the key of the scene and the diff of the means
        for k, v in prod_bins.items():

            means_scene_k = scene_bins.get(k)

            # bounding box correct if the difference of the bins in the 3 channels,
            # are below a certain threshold
            r_diff = np.absolute(v[0] - means_scene_k[0])
            g_diff = np.absolute(v[1] - means_scene_k[1])
            b_diff = np.absolute(v[2] - means_scene_k[2])

            if r_diff >= DIFF_COLOR_CHANNEL or g_diff >= DIFF_COLOR_CHANNEL or b_diff >= DIFF_COLOR_CHANNEL:

                good = False


        if good:

            corners_solution.append([ (int(box_top_left[0]), int(box_top_left[1])),
                                    (int(box_bot_right[0]), int(box_bot_right[1]))])


    return corners_solution

"""Function that solve the problem of 2 overlapping boxes"""

# function that merges adjacent corners that overlap each other
def merge_overlapping_corners(corners, DISTANCE_BOX = 200):

    corners_final = []

    for index1, [corner_top_left, corner_bot_right] in enumerate(corners):

        #add first couple of top left and bot right corners
        if len(corners_final) == 0:

            corners_final.append([ (int(corner_top_left[0]), int(corner_top_left[1])),
                                     (int(corner_bot_right[0]), int(corner_bot_right[1]))])


        for index2, [corner_fin_top_left, corner_fin_bot_right] in enumerate(corners_final):

            #if a corner is already in the final corners I don't add it
            if corner_top_left == corner_fin_top_left and corner_bot_right == corner_fin_bot_right:

                break

            # if a corner is near a final corner, then I do the mean of the two
            if ( distance_2points(corner_top_left, corner_fin_top_left) < DISTANCE_BOX and
            distance_2points(corner_bot_right, corner_fin_bot_right) < DISTANCE_BOX):

                top_left_sum = tuple(map(operator.add, corner_top_left, corner_fin_top_left))
                bot_right_sum = tuple(map(operator.add, corner_bot_right, corner_fin_bot_right))
                top_left_mean = (top_left_sum[0]/2, top_left_sum[1]/2)
                bot_right_mean = (bot_right_sum[0]/2, bot_right_sum[1]/2)

                corners_final[index2] = [ (int(top_left_mean[0]), int(top_left_mean[1])),
                                          (int(bot_right_mean[0]), int(bot_right_mean[1]))]

                break

            # if the corner isn't near any final corner then I add it to the final corners
            if index2 == len(corners_final) - 1:

                corners_final.append([ (int(corner_top_left[0]), int(corner_top_left[1])),
                                     (int(corner_bot_right[0]), int(corner_bot_right[1]))])

    return corners_final

"""Function that plot the box in the scene"""

# function that plot the final merged bounding boxes
def plot_box_corners(img, corners, scene, thickness = 10):

    if img is None:

        img = np.copy(scene)

    # print final corner
    for corner_top_left, corner_bot_right in corners:

        # compute width and height of the box
        corner_top_right = (corner_bot_right[0], corner_top_left[1])
        corner_width = distance_2points(corner_top_left, corner_top_right)
        corner_height = distance_2points(corner_top_right, corner_bot_right)

        # control that the box has a cereal box shape
        if corner_width < corner_height:

            img_scene_corners = cv2.rectangle(img, corner_top_left, corner_bot_right, (0, 255, 0), thickness)

        else:

            return None


    if len(corners) > 0:

        return img_scene_corners

    else:

        return None

"""- Load the product image
- Load the scene image
"""

#read the product image

#set the dir for product img and their name, dictionary with images, keypoints
#and descriptor
dirname_prod = 'models/'
imgs_prod = ['0', '1', '11', '19', '24', '25', '26']
img_prod = {}

INDEX_IMAGE = 0
INDEX_KP = 1
INDEX_DES = 2
INDEX_CENTER = 3
INDEX_V = 4

#save every image, keypoints and descriptor in the dictionary
for i in imgs_prod:

    img = cv2.imread(dirname + dirname_prod + i + '.jpg', cv2.COLOR_BGR2RGB)
    kp_prod = sift.detect(img)
    kp_prod, des_prod = sift.compute(img, kp_prod)
    img_prod[i] = [img, kp_prod, des_prod, 0]
    FindCenter(img_prod, i, INDEX_CENTER)

#read the scene image

#set the dir for test img and their name, prepare the dictionary with images,
#keypoints and descriptor
dirname_scene = 'scenes/'
imgs_scene = ['1', '2', '3', '4', '5']
img_scene = {}

#save every images, keypoints and descriptor
for j in imgs_scene:

    img = cv2.imread(dirname + dirname_scene + 'm' + j + '.png', cv2.COLOR_BGR2RGB)
    kp_scene = sift.detect(img)
    kp_scene, des_scene = sift.compute(img, kp_scene)
    img_scene[j] = [img, kp_scene, des_scene]

final_results = {}
final_scene = {}

# Set parameters

# minumum vote to consider C as a valid point
MIN_VOTE = 1

# dimension of a single cell of the accumulator array
DIMENSION1_CELL = 120
DIMENSION2_CELL = 120

# threshold for the object retrieve function
THRESHOLD = 0.5
min_match_count = 10
# 0.5 15
# distance to merge 2 corners
DISTANCE_BOX = 200

DIFF_COLOR = 82

BINS_WIDTH = 3
BINS_HEIGHT = 4


for j in imgs_scene:

    final_scene[j] = np.copy(img_scene[j][INDEX_IMAGE])
    final_results[j] = {}

    for i in imgs_prod:

        final_results[j][i] = {}
        final_results[j][i]['count'] = 0

        matches = object_retrieve(img_prod[i][INDEX_DES], img_scene[j][INDEX_DES], threshold=THRESHOLD)

        if len(matches) >= min_match_count:

            V = VoteVectors(matches, img_prod, i)

            Acc, C_scene, Acc_points = Array_acc(DIMENSION1_CELL,
                                                DIMENSION2_CELL,
                                                img_scene[j],
                                                img_prod[i],
                                                matches,
                                                V,
                                                i, j)




            #list of indexes of cells with num_votes >= min_vote
            accepted_cells = []

            for h in range(Acc.shape[0]):

                for w in range(Acc.shape[1]):

                    if Acc[h, w] >= MIN_VOTE:

                        accepted_cells.append([h, w])


            #estimated center
            C_scene, r = estimate_center(img_prod[i], img_scene[j], matches, V)

            # get the scaled height and width of the product in the scene
            prod_height = img_prod[i][INDEX_IMAGE].shape[0] / r
            prod_width = img_prod[i][INDEX_IMAGE].shape[1] / r

            # I consider the corners good if it has a shape of a rectangle with height > width like a cereal box
            if prod_width < prod_height:

                corners = []

                for c in accepted_cells:

                    # compute the C as the mean of points that are in the highleted cell
                    C_mean = np.mean(Acc_points[(c[0], c[1], 'S')], axis=0)

                    box_top_left = ( int(int(C_mean[0]) - (prod_width / 2)),
                                    int(int(C_mean[1]) - (prod_height / 2)))

                    box_bot_right = ( int(int(C_mean[0]) + (prod_width / 2)),
                                    int(int(C_mean[1]) + (prod_width / 2)))

                    corners.append([box_top_left, box_bot_right])



                #correct exceeding dimension
                box_corners_no_exceeding = correct_box_exceeding(corners, img_scene[j][INDEX_IMAGE])

                #solve color problem
                box_corners_no_color_problem = correct_color_problem(
                    box_corners_no_exceeding,
                    img_scene,
                    i, j, N = BINS_WIDTH, M = BINS_HEIGHT,
                    DIFF_COLOR_CHANNEL = DIFF_COLOR)

                # merge overlapping corners
                box_final_corners = merge_overlapping_corners(box_corners_no_color_problem,
                                                             DISTANCE_BOX)

                # plot the final corners
                img_corner = plot_box_corners(final_scene[j],
                                             box_final_corners,
                                             img_scene[j][INDEX_IMAGE])

                # store the final img and corners
                if img_corner is not None:

                    final_scene[j] = img_corner

                final_results[j][i]['count'] += len(box_final_corners)

                for index, [corner_top_left, corner_bot_right] in enumerate(box_final_corners):

                    corner_top_right = (corner_bot_right[0], corner_top_left[1])

                    if index == 0:
                        final_results[j][i]['width'] = []
                        final_results[j][i]['height'] = []
                        final_results[j][i]['pos'] = []

                    # compute width and height of the box
                    corner_width = distance_2points(corner_top_left, corner_top_right)
                    corner_height = distance_2points(corner_top_right, corner_bot_right)

                    # if after all the modification the box has no more cereal box shape I delete it
                    if corner_width < corner_height:

                        final_results[j][i]['count'] -= len(box_final_corners)
                        # save width, height and position of corner
                        final_results[j][i]['width'].append(corner_width)
                        final_results[j][i]['height'].append(corner_height)
                        final_results[j][i]['pos'].append((int(corner_top_left[0]), int(corner_top_left[1])))

"""Final output of Step B"""

for j in img_scene:

    plt.imshow(cv2.cvtColor(final_scene[j], cv2.COLOR_BGR2RGB))
    plt.show()
    print_result(final_results, j)

"""## Step C - Whole Shelve Challenge

Function to find the shelves on the scene and their coordinates
"""

# get the horizontal line of the image (used for the shelves later)
def get_horizontal_line(index, scene):

    if len(scene.shape) != 2:

        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    else:

        gray = scene

    # apply adaptive Threshold at the bitwise_not of gray
    gray = cv2.bitwise_not(gray)
    horizontal = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 20

    horizontal_Structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 2))

    horizontal = cv2.erode(horizontal, horizontal_Structure)
    horizontal = cv2.erode(horizontal, horizontal_Structure)

    horizontal = cv2.dilate(horizontal, horizontal_Structure)
    horizontal = cv2.dilate(horizontal, horizontal_Structure)
    horizontal = cv2.dilate(horizontal, horizontal_Structure)
    horizontal = cv2.dilate(horizontal, horizontal_Structure)
    horizontal = cv2.dilate(horizontal, horizontal_Structure)
    horizontal = cv2.dilate(horizontal, horizontal_Structure)
    horizontal = cv2.dilate(horizontal, horizontal_Structure)
    horizontal = cv2.dilate(horizontal, horizontal_Structure)

    return horizontal

# store the height coord of the shelves
def set_height_coord_of_shelves(shelves, horizontal, j):

    hor_height = horizontal.shape[0]
    hor_width = horizontal.shape[1]

    # image with horizontal feature in wich shelves are red
    hor_shelves_lines = np.copy(cv2.cvtColor(horizontal, cv2.COLOR_BGR2RGB))

    # scene img with red shelves
    scene_shelves_lines = np.copy(img_scene[j][INDEX_IMAGE])

    shelves.setdefault(j, []).append(0)

    # cycle the pixels on the height of the image
    for h in range(hor_height):

        # check in the horizontal image that intensities are 0 or 255
        # if there is a white pixel and if the previous pixel was black then record the position of a shelf

        if (horizontal[h, MEASURE_WIDTH] == 255 and horizontal[h - 1, MEASURE_WIDTH] == 0):

            shelves.setdefault(j, []).append(h)
            hor_shelves_lines = cv2.line(hor_shelves_lines, (0, h), (hor_width, h), (0, 0, 255), 3)

            scene_shelves_lines = cv2.line(scene_shelves_lines, (0, h), (hor_width, h), (0, 0, 255), 3)

# get the shelves in the scene image
def get_shelves_from_scene(index, scene_shelves, shelves, scene):

    MIN_HEIGHT_SHELVES = 20

    h_scene, w_scene, = scene.shape[:2]

    for i, h in enumerate(shelves[index]):

        if i == 0:

            scene_shelves[index] = {}

        # with the last shelf I have to take the portion of image from its height coord to the height of the image
        if i == len(shelves[index]) - 1:

            # good shelf only if its height >= threshold
            if h_scene - h >= MIN_HEIGHT_SHELVES:

                scene_shelves[index][i] = [scene[h:h_scene][0:w_scene]]

        # else from height coord to the next height coord
        else:

            next_h_coord = shelves[index][i+1]

            if next_h_coord - h >= MIN_HEIGHT_SHELVES:

                scene_shelves[index][i] = [scene[h:next_h_coord][0:w_scene]]

"""Function that are slightly different from the previous one"""

#function to solve the exceeding dimension
def correct_box_exceeding(corners, scene, box_conf, box_conf_exceeding, index):

    corners_no_exceeding = []

    #adjust overdimension
    scene_height = scene[INDEX_IMAGE].shape[0]
    scene_width = scene[INDEX_IMAGE].shape[1]

    for index, [corner_top_left, corner_bot_right] in enumerate(corners):

        prev_corner_top_left = corner_top_left
        prev_corner_bot_right = corner_bot_right

        corner_top_left = list(corner_top_left)
        corner_bot_right = list(corner_bot_right)

        #corners that goes out on the left of the image
        if corner_top_left[0] < 0:
            corner_top_left[0] = 0

        if corner_bot_right[0] < 0:
            corner_bot_right[0] = 0

        #corners that goes out on the right of the image
        if corner_top_left[0] > scene_width:
            corner_top_left[0] = scene_width

        if corner_bot_right[0] > scene_width:
            corner_bot_right[0] = scene_width

        #corners that goes out on the top of the image
        if corner_top_left[1] < 0:
            corner_top_left[1] = 0

        if corner_bot_right[1] < 0:
            corner_bot_right[1] = 0

        #corners that goes out on the bot of the image
        if corner_top_left[1] > scene_height:
            corner_top_left[1] = scene_height

        if corner_bot_right[1] > scene_height:
            corner_bot_right[1] = scene_height


        corner_top_left = tuple(corner_top_left)
        corner_bot_right = tuple(corner_bot_right)

        corners_no_exceeding.append([ (int(corner_top_left[0]),
                                      int(corner_top_left[1])),
                                      (int(corner_bot_right[0]),
                                      int(corner_bot_right[1]))])

        box_conf_exceeding[ (int(corner_top_left[0]), int(corner_top_left[1])),
                           (int(corner_bot_right[0]), int(corner_bot_right[1]))] = \
                            box_conf[(prev_corner_top_left, prev_corner_bot_right)]



    return corners_no_exceeding

# create and array with the position of the center
def Array_acc(DIMENSION1_CELL, DIMENSION2_CELL, scene, prod, img_prod, img_scene, matches, V, i, j):

    # set Accumulator

    Acc_dim = (int(img_scene[0].shape[0] / DIMENSION2_CELL),
            int(img_scene[0].shape[1] / DIMENSION1_CELL))


    Acc_points = {}

    Acc_array = np.zeros(Acc_dim)

    #get keypoints of the scene that are good matches
    pts_scene = np.float32([img_scene[INDEX_KP][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    #compute the mean scale factor and the mean angle between product and scene
    r, alpha = compare_keypoints(img_prod, img_scene, matches)

    # list with C estimation
    scene_C = []

    for m in range(len(matches)):

        #rescaling Vote Vector
        scene_Vx = V[m][0] / r
        scene_Vy = V[m][1] / r

        #compute estimated position of the center
        scene_Cx = pts_scene[m][0][0] - scene_Vx
        scene_Cy = pts_scene[m][0][1] - scene_Vy

        # store the estimation
        scene_C.append([scene_Cx, scene_Cy])

        C_x = int(scene_Cx / (DIMENSION1_CELL))
        C_y = int(scene_Cy / (DIMENSION2_CELL))

        if C_x in range(Acc_dim[1]):
            if C_y in range(Acc_dim[0]):

                Acc_array[C_y, C_x] += 1

                # save the scene point
                if not(C_y, C_x, 'S') in Acc_points:
                    Acc_points[(C_y, C_x, 'S')] = []
                    Acc_points[(C_y, C_x, 'S')].append((scene_Cx, scene_Cy))

                else:

                    Acc_points[(C_y, C_x, 'S')].append((scene_Cx, scene_Cy))


    return Acc_array, scene_C, Acc_points

# compare the kypoint and the angle and define the mean scale factor and the mena relative angle
def compare_keypoints(prod, scene, matches):

    size_prod_kp = [] #keypoints product
    size_scene_kp = [] #keypoints scene
    size_ratio = [] # ratio keypoints product and scene

    angle_prod_kp = [] # matched keypoint product
    angle_scene_kp = [] #matched keypoint scene
    angle_relatives = [] #rotation between product and scene

    for m in matches:

        # 2 variables with the size of the keypoints
        size_kp_P = np.float32(prod[i][INDEX_KP][m.queryIdx].size)
        size_kp_S = np.float32(scene[INDEX_KP][m.trainIdx].size)


        size_prod_kp.append(size_kp_P)
        size_scene_kp.append(size_kp_S)

        # 2 variables with the angle of the keypoints
        angle_kp_P = np.float32(prod[i][INDEX_KP][m.queryIdx].angle)
        angle_kp_S = np.float32(scene[INDEX_KP][m.trainIdx].angle)

        angle_prod_kp.append(angle_kp_P)
        angle_scene_kp.append(angle_kp_S)

        # define change in dimension between kp of the product and the scene
        size_ratio.append(size_kp_P / size_kp_S)

        # define change in dimension between kp of the produc and scene
        angle_relatives.append(angle_kp_P - angle_kp_S)


    # compute the Mean scale factor
    if len(size_ratio):
        Mean_Scale_Factor = (sum(size_ratio)) / len(size_ratio)

    else:
        Mean_Scale_Factor = 3.5

    # compute the mean relative angle
    if len(angle_relatives):
        Mean_Relative_Angle = (sum(angle_relatives)) / len(angle_relatives)

    else:
        Mean_Relative_Angle = 0


    return Mean_Scale_Factor, Mean_Relative_Angle

# function to solve the color problem in the image
def correct_color_problem(corners, scene, box_conf_exceeding, box_conf_color, i, j, N=3, M=4, DIFF_COLOR_CHANNEL = 50,
                          MAX_NO_GOOD_CELLS = 5):

    corners_color = []

    for index, [corner_top_left, corner_bot_right] in enumerate (corners):

        prod_bins = image_in_bins(img_prod[i][INDEX_IMAGE], N, M)

        scene_bins = image_in_bins(scene[INDEX_IMAGE][corner_top_left[1]: corner_bot_right[1],
                                                          corner_top_left[0]: corner_bot_right[0]], N, M)


        if not prod_bins or not scene_bins:
            return []

        good = True
        num_no_good = 0


        #cycle the prod dictionary, get the key of the scene and the diff of the means
        for k, v in prod_bins.items():

            if k[0] != 0 and k[0] != N:

                means_scene_k = scene_bins.get(k)

                # bounding box correct if the difference of the bins in the 3 channels,
                # are below a certain threshold
                r_diff = np.absolute(v[0] - means_scene_k[0])
                g_diff = np.absolute(v[1] - means_scene_k[1])
                b_diff = np.absolute(v[2] - means_scene_k[2])

                if r_diff >= DIFF_COLOR_CHANNEL or g_diff >= DIFF_COLOR_CHANNEL or b_diff >= DIFF_COLOR_CHANNEL:

                    good = False
                    num_no_good += 1


        # if the cells with too much diff are <= MAX number of no good cells I save them
        if num_no_good <= MAX_NO_GOOD_CELLS:

            corners_color.append([ (int(corner_top_left[0]), int(corner_top_left[1])),
                                    (int(corner_bot_right[0]), int(corner_bot_right[1]))])

            conf_color = (1 - (num_no_good / MAX_NO_GOOD_CELLS))

            box_conf_color[(int(corner_top_left[0]), int(corner_top_left[1])),
                          (int(corner_bot_right[0]), int(corner_bot_right[1]))] = \
                        box_conf_exceeding[(corner_top_left, corner_bot_right)]

            box_conf_color[(int(corner_top_left[0]), int(corner_top_left[1])),
                          (int(corner_bot_right[0]), int(corner_bot_right[1]))].append(conf_color)


    return corners_color

# function that merges adjacent corners that overlap each other
def merge_overlapping_corners(corners, box_conf_hsv, box_conf_final, i, j, sh_index, DISTANCE_BOX = 200):

    corners_final = []

    for index1, [corner_top_left, corner_bot_right] in enumerate(corners):

        #add first couple of top left and bot right corners
        if len(corners_final) == 0:

            corners_final.append([ (int(corner_top_left[0]), int(corner_top_left[1])),
                                     (int(corner_bot_right[0]), int(corner_bot_right[1]))])

            box_conf_final[(int(corner_top_left[0]), int(corner_top_left[1])),
                          (int(corner_bot_right[0]), int(corner_bot_right[1])), i, j, sh_index] =\
                            box_conf_hsv[(int(corner_top_left[0]), int(corner_top_left[1])),
                                        (int(corner_bot_right[0]), int(corner_bot_right[1]))]

            box_conf_final[(int(corner_top_left[0]), int(corner_top_left[1])),
                          (int(corner_bot_right[0]), int(corner_bot_right[1])), i, j, sh_index].append((i, j, sh_index))



        for index2, [corner_fin_top_left, corner_fin_bot_right] in enumerate(corners_final):

            #if a corner is already in the final corners I don't add it
            if corner_top_left == corner_fin_top_left and corner_bot_right == corner_fin_bot_right:

                break

            # if a corner is near a final corner, then I do the mean of the two
            if ( distance_2points(corner_top_left, corner_fin_top_left) < DISTANCE_BOX and
            distance_2points(corner_bot_right, corner_fin_bot_right) < DISTANCE_BOX):

                top_left_sum = tuple(map(operator.add, corner_top_left, corner_fin_top_left))
                bot_right_sum = tuple(map(operator.add, corner_bot_right, corner_fin_bot_right))
                top_left_mean = (top_left_sum[0]/2, top_left_sum[1]/2)
                bot_right_mean = (bot_right_sum[0]/2, bot_right_sum[1]/2)

                corners_final[index2] = [ (int(top_left_mean[0]), int(top_left_mean[1])),
                                          (int(bot_right_mean[0]), int(bot_right_mean[1]))]

                # keep track of confidences
                box_conf_final.pop((corner_fin_top_left, corner_fin_bot_right, i, j, sh_index), None)

                corner = ( (int(corner_fin_top_left[0]), int(corner_fin_top_left[1])),
                         (int(corner_fin_bot_right[0]), int(corner_fin_bot_right[1])))

                corner_nearest = nearest_corner(corner, box_conf_hsv)

                box_conf_final[(int(top_left_mean[0]), int(top_left_mean[1])),
                              (int(bot_right_mean[0]), int(bot_right_mean[1])), i, j, sh_index] = \
                                box_conf_hsv[corner_nearest]

                box_conf_final[(int(top_left_mean[0]), int(top_left_mean[1])),
                              (int(bot_right_mean[0]), int(bot_right_mean[1])), i, j, sh_index].append((i, j, sh_index))


                break

            # if the corner isn't near any final corner then I add it to the final corners
            if index2 == len(corners_final) - 1:

                corners_final.append([ (int(corner_top_left[0]), int(corner_top_left[1])),
                                     (int(corner_bot_right[0]), int(corner_bot_right[1]))])

                box_conf_final[corner_top_left, corner_bot_right, i, j, sh_index] = \
                                    box_conf_hsv[(corner_top_left, corner_bot_right)]

                box_conf_final[corner_top_left, corner_bot_right, i, j, sh_index].append((i, j, sh_index))



    return corners_final

"""Function that find the nearest corner in the dictionary"""

# find the nearest corner in the dictionary
def nearest_corner(corner, box_conf_hsv):

    i = 0

    for top_left, bot_right in box_conf_hsv.keys():

        if i == 0:

            nearest = (top_left, bot_right)
            dist_min = distance_2points(top_left, corner[0])
            i = 1

        dist_min_tmp = distance_2points(top_left, corner[0])

        if dist_min_tmp < dist_min:

            nearest = (top_left, bot_right)
            dist_min = dist_min_tmp

    return nearest

"""Function that correct the HSV correlation"""

# function that correct the HSV correlation
def correct_HSV_correlation(corners, scene, i, j, box_conf_exceeding, box_conf_hsv, cut_factor, MIN_HIST_CORR):

    corner_color = []

    prod_img = np.copy(img_prod[i][INDEX_IMAGE])

    for index, [corner_top_left, corner_bot_right] in enumerate(corners):

        prod_in_scene = np.copy(scene[INDEX_IMAGE][corner_top_left[1]:corner_bot_right[1],
                                                  corner_top_left[0]:corner_bot_right[0]])


        # define the new dimension for the product that is the probability that the product is in the scene
        new_dim = (prod_in_scene.shape[1], prod_in_scene.shape[0])

        # product image resizing
        prod_resized = cv2.resize(prod_img, new_dim, interpolation = cv2.INTER_AREA)
        prod_resized = prod_resized.reshape(prod_in_scene.shape[0], prod_in_scene.shape[1], prod_in_scene.shape[2])

        height_prod_resized, width_prod_resized, = prod_resized.shape[:2]
        height_prod_in_scene, width_prod_in_scene = prod_in_scene.shape[:2]

        # cut top and bot image
        prod_resized = prod_resized[int(height_prod_resized / cut_factor): height_prod_resized - int(height_prod_resized / cut_factor),
                                        0: width_prod_in_scene]

        prod_in_scene = prod_in_scene[int(height_prod_in_scene / cut_factor): height_prod_in_scene - int(height_prod_in_scene / cut_factor),
                                     0: width_prod_in_scene]

        #get the Hist Correlation and if it's >= Minimum HIST correlation save it
        hist_corr = getCorrelationHist(prod_resized, prod_in_scene)

        hist_hsv_corr_conf = (hist_corr / MIN_HIST_CORR)
        box_conf_hsv[(corner_top_left, corner_bot_right)] = box_conf_exceeding[(corner_top_left, corner_bot_right)]
        box_conf_hsv[(corner_top_left, corner_bot_right)].append(hist_hsv_corr_conf)

        if hist_corr >= MIN_HIST_CORR:

            corner_color.append([(int(corner_top_left[0]), int(corner_top_left[1])),
                                (int(corner_bot_right[0]), int(corner_bot_right[1]))])


    return corner_color

"""Function to get the correlation Hist of 2 images"""

# function that compute the histogram of the 2 image and compare them
def getCorrelationHist(img1, img2):

    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    # hue varies from 0 to 179, saturation from 0 to 255
    h_range = [0, 180]
    s_range = [0, 256]
    ranges = h_range + s_range

    channels = [0, 1]

    img1_hist = cv2.calcHist([img1_hsv], channels, None, histSize, ranges, accumulate = False)
    cv2.normalize(img1_hist, img1_hist, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)

    img2_hist = cv2.calcHist([img2_hsv], channels, None, histSize, ranges, accumulate = False)
    cv2.normalize(img2_hist, img2_hist, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)

    return cv2.compareHist(img1_hist, img2_hist, 2) * 10

"""- Load the product file
- Load the scene file
"""

#read the product image

#set the dir for product img and their name, dictionary with images, keypoints
#and descriptor
dirname_prod = 'models/'
num_prod = 24
img_prod = {}
imgs_prod = list(range(1, 24))

INDEX_IMAGE = 0
INDEX_KP = 1
INDEX_DES = 2
INDEX_CENTER = 3
INDEX_V = 4

#save every image, keypoints and descriptor in the dictionary
for i in range(num_prod):

    img = cv2.imread(dirname + dirname_prod + '{}.jpg'.format(i), cv2.COLOR_BGR2RGB)
    img_blurred = cv2.GaussianBlur(img, (15, 15), 0)
    kp_prod = sift.detect(img_blurred)
    kp_prod, des_prod = sift.compute(img_blurred, kp_prod)
    img_prod[i] = [img_blurred, kp_prod, des_prod, 0]
    FindCenter(img_prod, i, INDEX_CENTER)

#read the scene image

#set the dir for test img and their name, prepare the dictionary with images,
#keypoints and descriptor
dirname_scene = 'scenes/'
imgs_scene = ['1', '2', '3', '4', '5']
img_scene = {}

#save every images, keypoints and descriptor
for j in imgs_scene:

    img = cv2.imread(dirname + dirname_scene + 'h' + j + '.jpg', cv2.COLOR_BGR2RGB)
    kp_scene = sift.detect(img)
    kp_scene, des_scene = sift.compute(img, kp_scene)
    img_scene[j] = [img, kp_scene, des_scene]

"""Find and save the shelves on the scene, then compute the keypoints and the descriptors"""

MEASURE_WIDTH = 100

shelves_height_coord = {}
scene_shelves = {}

for j in  imgs_scene:

    img_horizontal = get_horizontal_line(j, img_scene[j][INDEX_IMAGE])

    set_height_coord_of_shelves(shelves_height_coord, img_horizontal, j)

    get_shelves_from_scene(j, scene_shelves, shelves_height_coord, img_scene[j][INDEX_IMAGE])

for sc_index, shelves_of_scene in scene_shelves.items():

    for sh_index, sh_feat in shelves_of_scene.items():

        # save the image of a single shelf in the structure
        # compute the keypoints and descriptors and save them too
        kp_sh = sift.detect(sh_feat[INDEX_IMAGE])
        kp_sh, des_sh = sift.compute(sh_feat[INDEX_IMAGE], kp_sh)
        shelves_of_scene[sh_index] = [sh_feat[INDEX_IMAGE], kp_sh, des_sh]

"""Code to find the product and their corners in the scene"""

final_results = {}
final_scene = {}

min_votes_confidence = {}
box_confidence = {}
box_conf_exceeding = {}
box_conf_color = {}
box_conf_hsv = {}
box_conf_final = {}


# Set parameters

# minumum vote to consider C as a valid point
MIN_VOTES = 1

# dimension of a single cell of the accumulator array
DIMENSION1_CELL = 30
DIMENSION2_CELL = 30

# distance to merge 2 corners
DISTANCE_BOX = 50

DIFF_COLOR = 145

BINS_WIDTH = 3
BINS_HEIGHT = 4

# if the number of no good cell is <= of this, then I take the box
MAX_NO_GOOD = 3

# threshold for oject retrieve
THRESHOLD = 0.65

# minumum threshold of correlation of hsv histogram
min_hsv_corr = [169, 71.5, 105, 195, 68, 70, 76, 111, 26, 11, 18, 59, 14, 14, 70, 75, 150, 57, 60, 43, 31, 22, 115, 10.5, 70, 150, 64]

# amount of cutting on top and bottom of image when considering hsv Hist Correlation
CUT_FACTOR_HIST = 7

for j in imgs_scene:

    final_results[j] = {}
    final_scene[j] = np.copy(img_scene[j][INDEX_IMAGE])

    # cycle shelves in scene
    for sh_idx, sh_feat in scene_shelves[j].items():

        # cycle product
        for i in imgs_prod:

            final_results[j][i] = {}
            final_results[j][i]['count'] = 0
            final_results[j][i]['width'] = []
            final_results[j][i]['height'] = []
            final_results[j][i]['pos'] = []
            final_results[j][i]['conf_value'] = []
            final_results[j][i]['final_conf'] = []

            prod = np.copy(img_prod[i][INDEX_IMAGE])

            h_prod, w_prod = prod.shape[:2]

            r_prod = h_prod / w_prod

            img_shelf = np.copy(sh_feat[INDEX_IMAGE])

            h_shelf, w_shelf = img_shelf.shape[:2]

            matches = object_retrieve(img_prod[i][INDEX_DES], sh_feat[INDEX_DES], threshold = THRESHOLD)

            V = VoteVectors(matches, img_prod, i)

            Acc, C_scene, Acc_points = Array_acc(DIMENSION1_CELL,
                                            DIMENSION2_CELL,
                                            img_shelf,
                                            prod,
                                            img_prod,
                                            sh_feat,
                                            matches,
                                            V,
                                            i, j)


            accepted_cells = []

            for h in range(Acc.shape[0]):
                for w in range(Acc.shape[1]):

                    if Acc[h, w] >= MIN_VOTES:

                        accepted_cells.append([h, w])
                        min_votes_confidence[(h, w)] = (Acc[h, w] / MIN_VOTES)

            C_scene, r = estimate_center(img_prod, sh_feat, matches, V)

            h_prod_scene = img_prod[i][INDEX_IMAGE].shape[0] / r
            w_prod_scene = img_prod[i][INDEX_IMAGE].shape[1] / r

            corners = []

            for c in accepted_cells:

                # Compute the center as the mean of points in the highlighted cell
                C_mean = np.mean(Acc_points[(c[0], c[1], 'S')], axis = 0)

                # Compute all the corner of the box
                corner_top_left = ( int( int(C_mean[0]) - (w_prod_scene / 2) ),
                                   int( int(C_mean[1]) - (h_prod_scene / 2)))

                corner_bot_right = ( int( int(C_mean[0]) + (w_prod_scene / 2)),
                                  int ( int(C_mean[1]) + (h_prod_scene / 2)))

                corner_top_right = (corner_bot_right[0], corner_top_left[1])

                corner_bot_left = (corner_top_left[0], corner_bot_right[1])

                h_corner = distance_2points(corner_top_left, corner_bot_left)

                w_corner = distance_2points(corner_top_left, corner_top_right)

                # consider box good only if it doesn't exceed WIDTH_MAX and HEIGHT_MAX
                height_max = h_shelf
                width_max = height_max / r_prod

                if h_corner <= height_max and w_corner <= width_max:

                    corners.append([corner_top_left, corner_bot_right])

                    box_confidence[(corner_top_left, corner_bot_right)] = [min_votes_confidence[(c[0], c[1])]]


            #correct exceeding dimension
            box_corners_no_exceeding = correct_box_exceeding(corners, sh_feat, box_confidence, box_conf_exceeding, j)

            #solve color problem
            box_corners_no_color_problem = correct_color_problem(box_corners_no_exceeding, sh_feat, box_conf_exceeding,
                                                                box_conf_color, i, j, N = BINS_WIDTH, M= BINS_HEIGHT,
                                                                DIFF_COLOR_CHANNEL = DIFF_COLOR,
                                                                MAX_NO_GOOD_CELLS = MAX_NO_GOOD)

            #solve color problem with HSV
            box_corners_no_color_problem = correct_HSV_correlation(box_corners_no_color_problem, sh_feat, i, j,
                                                                  box_conf_color, box_conf_hsv, CUT_FACTOR_HIST,
                                                                  MIN_HIST_CORR = min_hsv_corr[i])

            # merge overlapping corners
            box_final_corners = merge_overlapping_corners(box_corners_no_color_problem, box_conf_color, box_conf_final,
                                                        i, j, sh_idx, DISTANCE_BOX)

"""After finding the principal box, it will be applayed 2 filters
- box have to exceed a certain threshold
- if 2 box are nearer than a certain threshold they are considered overlapped so only the one with greater confidence will be saved
"""

# I consider only the box with a confidence higher than a threshold
box_conf_threshold = {}

CONF_THRES = 5

for k, v in box_conf_final.items():

    confidence_value = 0.7 * v[0] + 5 * v[1] + 0.8 * v[2]
    box_conf_final[k].append(confidence_value)

    if confidence_value >= CONF_THRES:

        box_conf_threshold[k] = v

# list of excluded box due to a less confidence wrt another overlapped box
box_excluded = {}

# if the distance between 2 box is <= DISTANCE_THRESHOLD
# they are overlapped I will take only the one with higher confidence
DISTANCE_THRESHOLD = 50

for k, v in box_conf_threshold.items():

    corner_top_left_shelf = k[0]
    corner_bot_right_shelf = k[1]
    scene_idx = v[3][1]
    shelf_idx = v[3][2]
    confidence_value = v[-1]

    corner_top_left_scene = (corner_top_left_shelf[0], corner_top_left_shelf[1] + shelves_height_coord[scene_idx][shelf_idx])
    corner_bot_right_scene = (corner_bot_right_shelf[0], corner_bot_right_shelf[1] + shelves_height_coord[scene_idx][shelf_idx])

    for k1, v1 in box_conf_threshold.items():

        if k==k1:
            break

        corner_top_left_shelf1 = k1[0]
        corner_bot_right_shelf1 = k1[1]
        scene_idx1 = v1[3][1]
        shelf_idx1 = v1[3][2]
        confidence_value1 = v1[-1]

        corner_top_left_scene1 = (corner_top_left_shelf1[0],
                                  corner_top_left_shelf1[1] + shelves_height_coord[scene_idx1][shelf_idx1])
        corner_bot_right_scene1 = (corner_bot_right_shelf1[0],
                                   corner_bot_right_shelf1[1] + shelves_height_coord[scene_idx1][shelf_idx1])

        if distance_2points(corner_top_left_scene, corner_top_left_scene1) <= DISTANCE_THRESHOLD:

            if scene_idx == scene_idx1 and shelf_idx == shelf_idx1:

                if confidence_value <= confidence_value1:

                    box_excluded[k] = 1

                else:

                    box_excluded[k1] = 1

"""Save the  box that passed the 2 filter"""

# save the information of the corners that passed all the controls

for k, v in box_conf_final.items():

    if k not in box_excluded:

        corner_top_left_shelf = k[0]
        corner_bot_right_shelf = k[1]
        prod_idx = v[3][0]
        scene_idx = v[3][1]
        shelf_idx = v[3][2]

        weight_num_vote = v[0]
        weight_color = v[1]
        weight_hsv = v[2]
        confidence_value = v[-1]

        corner_top_left_scene = (corner_top_left_shelf[0], corner_top_left_shelf[1] + shelves_height_coord[scene_idx][shelf_idx])
        corner_bot_right_scene = (corner_bot_right_shelf[0], corner_bot_right_shelf[1] + shelves_height_coord[scene_idx][shelf_idx])

        final_scene[scene_idx] = cv2.rectangle(final_scene[scene_idx], corner_top_left_scene, corner_bot_right_scene,
                                              (0, 255, 0), 10)


        final_results[scene_idx][prod_idx]['count'] +=1

        corner_top_right_scene = (corner_bot_right_scene[0], corner_top_left_scene[1])

        width_box = distance_2points(corner_top_left_scene, corner_top_right_scene)
        height_box = distance_2points(corner_top_right_scene, corner_bot_right_scene)

         # save width, height, position, the three weighted values for final confidence and the final confidence of the box
        final_results[scene_idx][prod_idx]['width'].append(width_box)
        final_results[scene_idx][prod_idx]['height'].append(height_box)

        final_results[scene_idx][prod_idx]['pos'].append(
            (int(corner_top_left_scene[0]), int(corner_top_left_scene[1])))

        final_results[scene_idx][prod_idx]['conf_value'].append([weight_num_vote, weight_color, weight_hsv])

        final_results[scene_idx][prod_idx]['final_conf'].append(confidence_value)

"""Final output of Step C"""

for j in imgs_scene:

    plt.imshow(cv2.cvtColor(final_scene[j], cv2.COLOR_BGR2RGB))
    plt.show()
    print_result(final_results, j)
