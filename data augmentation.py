# HI JUST SOME NOTES TO PREVENT CRASHING
# run this script as it is the first time to see whats being done in the code
# when you are ready to generate and save the processed training images, comment out lines 190-200
# try to run this script for only a few classes at a time (e.g. c0,c1,c2) by commenting out the respective lines below
# make sure to include the img folder (not the topmost folder!) from kaggle in this working directory

# setup
import cv2
import os
import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

# load the image directories
train_dir = r'.\imgs\train' # image folder

# get the list of jpegs from sub image class folders
c0_imgs = [fn for fn in os.listdir(f'{train_dir}/c0') if fn.endswith('.jpg')]
c1_imgs = [fn for fn in os.listdir(f'{train_dir}/c1') if fn.endswith('.jpg')]
c2_imgs = [fn for fn in os.listdir(f'{train_dir}/c2') if fn.endswith('.jpg')]
c3_imgs = [fn for fn in os.listdir(f'{train_dir}/c3') if fn.endswith('.jpg')]
c4_imgs = [fn for fn in os.listdir(f'{train_dir}/c4') if fn.endswith('.jpg')]
c5_imgs = [fn for fn in os.listdir(f'{train_dir}/c5') if fn.endswith('.jpg')]
c6_imgs = [fn for fn in os.listdir(f'{train_dir}/c6') if fn.endswith('.jpg')]
c7_imgs = [fn for fn in os.listdir(f'{train_dir}/c7') if fn.endswith('.jpg')]
c8_imgs = [fn for fn in os.listdir(f'{train_dir}/c8') if fn.endswith('.jpg')]
c9_imgs = [fn for fn in os.listdir(f'{train_dir}/c9') if fn.endswith('.jpg')]

# convert images to grayscale (black and white)
def grayscale(imglist, subclass, img_dir):
    '''
    inputs:
    -imglist is list of jpegs from subimage class folder
    -subclass is a string, the name of subimage class folder
    -img_dir is the highest directory of the images

    outputs the augmented images stored in a list
    '''
    out = []
    for i in tqdm(range(len(imglist))):
        img_path = os.path.join(f'{img_dir}'+subclass, imglist[i])
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append(img_gray)
    return out

c0_gray = grayscale(c0_imgs, '/c0/', train_dir)
# c1_gray = grayscale(c1_imgs, '/c1/', train_dir)
# c2_gray = grayscale(c2_imgs, '/c2/', train_dir)
# c3_gray = grayscale(c3_imgs, '/c3/', train_dir)
# c4_gray = grayscale(c4_imgs, '/c4/', train_dir)
# c5_gray = grayscale(c5_imgs, '/c5/', train_dir)
# c6_gray = grayscale(c6_imgs, '/c6/', train_dir)
# c7_gray = grayscale(c7_imgs, '/c7/', train_dir)
# c8_gray = grayscale(c8_imgs, '/c8/', train_dir)
# c9_gray = grayscale(c9_imgs, '/c9/', train_dir)

# convert images using histogram equalization
def histeq(imglist):
    '''
    inputs is list of image values (not the filename!)

    outputs the augmented images stored in a list
    '''
    out = []
    for i in tqdm(range(len(imglist))):
        img = imglist[i]
        img_eq = cv2.equalizeHist(img)
        out.append(img_eq)
    return out

c0_eq = histeq(c0_gray)
# c1_eq = histeq(c1_gray)
# c2_eq = histeq(c2_gray)
# c3_eq = histeq(c3_gray)
# c4_eq = histeq(c4_gray)
# c5_eq = histeq(c5_gray)
# c6_eq = histeq(c6_gray)
# c7_eq = histeq(c7_gray)
# c8_eq = histeq(c8_gray)
# c9_eq = histeq(c9_gray)

# mix images of a class
def mix(imglist):
    '''
    inputs is list of image values (not the filename!)

    outputs the augmented images stored in a list
    '''
    out = []
    for i in tqdm(range(len(imglist))):
        # original image
        img1 = imglist[i]
        # randomly select another image from this class
        index2 = np.random.randint(len(imglist))
        # second image to be used in mixing
        img2 = imglist[index2]
        # mix them
        # alpha and beta must sum to 1.0
        img_mixed = cv2.addWeighted(img1,0.8,img2,0.2,0.0) # img1*alpha + img2*beta + gamma 
        out.append(img_mixed)
    return out

c0_mix = mix(c0_eq)
# c1_mix = mix(c1_eq)
# c2_mix = mix(c2_eq)
# c3_mix = mix(c3_eq)
# c4_mix = mix(c4_eq)
# c5_mix = mix(c5_eq)
# c6_mix = mix(c6_eq)
# c7_mix = mix(c7_eq)
# c8_mix = mix(c8_eq)
# c9_mix = mix(c9_eq)

# randomly rotate images
# random horizonal shift/vertical shift images
# we do not carry out shearing of images (shearing means stretching the image non-uniformly so it becomes squashed)
# reference: https://github.com/vxy10/ImageAugmentation
def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img

def rotate_shift(imglist):
    '''
    inputs is list of image values (not the filename!)

    outputs the augmented images stored in a list
    '''
    out = []
    for i in tqdm(range(len(imglist))):
        # adjust the angle range of rotation and translation range here
        # angle_range, shear_range, trans_range
        img_transformed = transform_image(imglist[i],10,0,10) 
        out.append(img_transformed)
    return out

c0_final = rotate_shift(c0_mix)
# c1_final = rotate_shift(c1_mix)
# c2_final = rotate_shift(c2_mix)
# c3_final = rotate_shift(c3_mix)
# c4_final = rotate_shift(c4_mix)
# c5_final = rotate_shift(c5_mix)
# c6_final = rotate_shift(c6_mix)
# c7_final = rotate_shift(c7_mix)
# c8_final = rotate_shift(c8_mix)
# c9_final = rotate_shift(c9_mix)

# visually check image conversion
plt.figure(figsize=(20,10))
plt.subplot(141),plt.imshow(c0_gray[0], cmap='gray'),
plt.title('Grayscaled image'),plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(c0_eq[0], cmap='gray'),
plt.title('Hist Equalized image'),plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(c0_mix[0], cmap='gray'),
plt.title('Mixed image'),plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(c0_final[0], cmap='gray'),
plt.title('Rotated/shifted image'),plt.xticks([]), plt.yticks([])
plt.show()

# save the augmented images under imgs folder
def saveimg(imglist, img_namelist,subclass,dir):
    '''
    imglist are the image values (actual image themselves)
    img_namelist contains the original jpeg names
    dir is the folder to save to
    '''
    # save the current working directory
    cwd = os.getcwd()

    # create target working directory
    file_path = os.path.join(f'{dir}'+subclass)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # change to target directory
    os.chdir(file_path)
    # save images
    for i in tqdm(range(len(imglist))):
        filename = img_namelist[i]
        cv2.imwrite(filename,imglist[i])
    # change back to original working directory
    os.chdir(cwd)

target_folder = r'.\imgs\trainprocessed'
saveimg(c0_final,c0_imgs,'/c0/',target_folder)
# saveimg(c1_final,c1_imgs,'/c1/',target_folder)
# saveimg(c2_final,c2_imgs,'/c2/',target_folder)
# saveimg(c3_final,c3_imgs,'/c3/',target_folder)
# saveimg(c4_final,c4_imgs,'/c4/',target_folder)
# saveimg(c5_final,c5_imgs,'/c5/',target_folder)
# saveimg(c6_final,c6_imgs,'/c6/',target_folder)
# saveimg(c7_final,c7_imgs,'/c7/',target_folder)
# saveimg(c8_final,c8_imgs,'/c8/',target_folder)
# saveimg(c9_final,c9_imgs,'/c9/',target_folder)