import torch
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--color', type=str, default='color', help='gray or color')
opt = parser.parse_known_args()[0]

if opt.color == 'color':
    v_path = '/Data3/soyeon/2023_ATC/anomal/code/yolov5/runs/train/exp13/weights/best.pt'
    cvtcolor = cv2.COLOR_BGR2RGB

else:
    v_path = '/Data3/soyeon/2023_ATC/yolov5/runs/train/exp12/weights/best.pt'
    cvtcolor = cv2.COLOR_BGR2GRAY
        
obj_path = '/Data3/soyeon/2023_ATC/yolov5/runs/train/exp2/weights/best.pt'

cap = cv2.VideoCapture('/Data3/soyeon/2023_ATC/anomal/data/etc/CH001/[CH001] 20230707021030-20230707021159.avi')

model = torch.hub.load('ultralytics/yolov5', 'custom', path=v_path) 

if not cap.isOpened():
	print('fail')
 
idx = 0
while True:
    idx += 1
    _, frame = cap.read()
    
    if idx%5 != 0:
        continue
    
    frame = cv2.cvtColor(frame, cvtcolor)
    frame = cv2.resize(frame, (int(3840/5), int(2160/5)))
    results = model(frame)
    results.save(save_dir='runs/detect/exp2/', exist_ok=True)    
 