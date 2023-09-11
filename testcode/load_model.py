import torch
import cv2
import matplotlib.pyplot as plt

v_path = '/Data3/soyeon/2023_ATC/anomal/code/yolov5/runs/train/exp13/weights/best.pt'
obj_path = '/Data3/soyeon/2023_ATC/yolov5/runs/train/exp2/weights/best.pt'

cap = cv2.VideoCapture('/Data3/soyeon/2023_ATC/anomal/data/etc/CH003/[CH003] 20230707010206-20230707010408.avi')

model = torch.hub.load('ultralytics/yolov5', 'custom', path=v_path) 

if not cap.isOpened():
	print('fail')
 
idx = 0
while True:
    idx += 1
    _, frame = cap.read()
    
    if idx%5 != 0:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (int(3840/5), int(2160/5)))
    results = model(frame)
    results.save(save_dir='runs/detect/exp2/', exist_ok=True)    
 