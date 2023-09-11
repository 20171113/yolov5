"""
Get gray scale image from input original image
"""


import cv2
import numpy as np
import os, sys
import argparse

bg_path = '/Data3/soyeon/2023_ATC/data/etc/Image/CH001/20230707010000-20230707010129/34.jpg'
bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
img_size = (int(bg.shape[1]/10), int(bg.shape[0]/10))
bg = cv2.resize(bg, img_size)
img_path = '/Data3/soyeon/2023_ATC/data/etc/anomaly_scene/108.JPEG'


def get_blurred_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)

    bg_int = bg.astype(int)
    img_int = img.astype(int)
    
    diff = np.abs(np.subtract(bg_int, img_int) + bg.mean()).astype(np.uint8)
    
    denoise_diff = cv2.fastNlMeansDenoising(diff, None, 50, 3, 3)
    
    diff_filter = img.copy()
    lower = 75
    upper = 115
    
    diff_filter[(lower <= denoise_diff) & (denoise_diff <= upper)] = 0
    diff_filter[(denoise_diff < lower) | (upper < denoise_diff)] = 1

    filtered_img = 255 * diff_filter
    
    return filtered_img
    #cv2.imwrite('diff_filter.jpg', filtered_img)
    
    

# parser = argparse.ArgumentParser()
# parser.add_argument('--img-path', type=str, default=img_path, help='img path')
# opt = parser.parse_known_args()[0]
# get_blurred_img(opt.img_path)
