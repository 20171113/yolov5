import cv2
import numpy as np
import os, sys, glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, help='image path')
parser.add_argument('--save-dir', type=str, help='image path')
opt = parser.parse_known_args()[0]
path = opt.path
save_dir = opt.save_dir

images = glob.glob(os.path.join(path, '*.jpg'))

for image in images:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    save_path = image.replace('test_frame', save_dir)
    print(save_path)
    cv2.imwrite(save_path, img)