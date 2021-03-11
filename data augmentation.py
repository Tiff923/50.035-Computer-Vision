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
'''c1_gray = grayscale(c1_imgs, '/c1/', train_dir)
c2_gray = grayscale(c2_imgs, '/c2/', train_dir)
c3_gray = grayscale(c3_imgs, '/c3/', train_dir)
c4_gray = grayscale(c4_imgs, '/c4/', train_dir)
c5_gray = grayscale(c5_imgs, '/c5/', train_dir)
c6_gray = grayscale(c6_imgs, '/c6/', train_dir)
c7_gray = grayscale(c7_imgs, '/c7/', train_dir)
c8_gray = grayscale(c8_imgs, '/c8/', train_dir)
c9_gray = grayscale(c9_imgs, '/c9/', train_dir)'''

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
'''c1_eq = histeq(c1_gray)
c2_eq = histeq(c2_gray)
c3_eq = histeq(c3_gray)
c4_eq = histeq(c4_gray)
c5_eq = histeq(c5_gray)
c6_eq = histeq(c6_gray)
c7_eq = histeq(c7_gray)
c8_eq = histeq(c8_gray)
c9_eq = histeq(c9_gray)'''

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
'''c1_mix = mix(c1_eq)
c2_mix = mix(c2_eq)
c3_mix = mix(c3_eq)
c4_mix = mix(c4_eq)
c5_mix = mix(c5_eq)
c6_mix = mix(c6_eq)
c7_mix = mix(c7_eq)
c8_mix = mix(c8_eq)
c9_mix = mix(c9_eq)'''

# randomly rotate images


# random horizonal shift/vertical shift images


# visually check image conversion
plt.figure(figsize=(20,10))
plt.subplot(151),plt.imshow(c0_gray[0], cmap='gray'),
plt.title('Grayscaled image'),plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(c0_eq[0], cmap='gray'),
plt.title('Hist Equalized image'),plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow(c0_mix[0], cmap='gray'),
plt.title('Mixed image'),plt.xticks([]), plt.yticks([])
plt.show()

# save and output augmented images in another folder
''' remember to add the output folder name to the gitignore file before commit/pushing !! '''