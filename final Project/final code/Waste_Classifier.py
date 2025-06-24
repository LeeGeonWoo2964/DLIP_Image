import argparse
# from curses import COLOR_BLACK
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.backends.cudnn as cudnn
import cv2 as cv
import math
from operator import imod
import numpy as np
from itertools import *

from CNN_initialize_model import initialize_model
from CNN_set_parameter_requires_grad import set_parameter_requires_grad

from PIL import Image
import torch
from ultralytics import YOLO
from torchvision import transforms
from calflops import calculate_flops
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pretrained YOLO model
yolo_model = YOLO('Pretrained Weights/yolov8m_epoch100best.pt')

model_path = "./Pretrained Weights/resnet_final_0623.pth"

cnn_model = torch.load(model_path, map_location='cuda')
cnn_model.eval()

#############
#Experiment YOLO+CNN
##########
def Yolo_cnn(frame):

    CNN_frame=frame.copy()
    result_img=frame.copy()
    
    #YOLO=====================
    # Perform object detection on an image using the model
    # result = yolo_model.predict(source=frame, iou=0.6, conf=0.3)
    result = yolo_model.predict(source=frame, iou=0.6)
    yolo_result=result[0]

    img=yolo_result.plot()
    boxes = result[0].boxes
    # Get Rectangle Point
    box_point=result[0].boxes.xyxy.cpu().numpy() #(x1,y1,x2,y2)

    if boxes is None or len(boxes) == 0:
        return frame     

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    Crop_tensor=[]
    pre_center_x=[]
    pre_center_y=[]
    pre_center_x.append(0)
    pre_center_y.append(0)
    CNN_src=CNN_frame #Gray Scale

    for i in range(len(box_point)):
        dx=min(abs(x-int(((box_point[i,2])+(box_point[i,0]))/2)) for x in pre_center_x)
        dy=min(abs(y-int(((box_point[i,3])+(box_point[i,1]))/2)) for y in pre_center_y)
        if dx >=80 and dy >=80:
            pre_center_x.append(int(((box_point[i,2])+(box_point[i,0]))/2))
            pre_center_y.append(int(((box_point[i,3])+(box_point[i,1]))/2))
        # 1. 인식된 객체를 자르고 tensor로 바꿔 CNN에 넣을 수 있도록 처리
            Temp_Crop=CNN_src[int(box_point[i,1]):int(box_point[i,3]),int(box_point[i,0]):int(box_point[i,2])]
            if isinstance(Temp_Crop,np.ndarray):
                Temp_tensor=Image.fromarray(Temp_Crop.astype('uint8')).convert('RGB')
                Temp_tensor=transform(Temp_tensor)
                Crop_tensor.append(Temp_tensor)
    if not Crop_tensor:
        return frame

    CNN_input_batch=torch.stack(Crop_tensor).to(device)
    #YOLO=====================

    # Class
    classes = ['cardboard', 'glass','metal', 'plastic']
    #class마다 색깔 지정
    cls_color=[(255,0,0),(0,255,0),(0,0,255),(255,255,0)]

    # CNN
    with torch.no_grad():
        output = cnn_model(CNN_input_batch)
        _, predicted = torch.max(output, 1)
        for i, pred in enumerate(predicted):
            print(f"[{i}] Predicted class: {classes[pred.item()]}")
            probs=torch.softmax(output[i],dim=0)
            object_class=classes[pred.item()]
            object_color=cls_color[pred.item()]
            if probs[pred.item()].item()<=0.0:
                continue
            #src에 사각형 그리고 class 이름 및 로봇 팔이 잡을 지점 표시
            pt1=[int(box_point[i,0]),int(box_point[i,1])]
            pt2=[int(box_point[i,2]),int(box_point[i,3])]
            circle_x=int(((box_point[i,2])+(box_point[i,0]))/2)
            circle_y=int(((box_point[i,3])+(box_point[i,1]))/2)
            text_pt=(int(box_point[i,0])+20,int(box_point[i,1]+20))
            cv.rectangle(result_img, pt1, pt2, object_color, thickness=10, lineType=None, shift=None)
            cv.circle(result_img,(circle_x,circle_y),10,object_color,-1,cv.LINE_AA)
            cv.putText(result_img,f"({object_class} {probs[pred.item()].item():.2f})",text_pt,cv.FONT_HERSHEY_SIMPLEX,3,object_color,5,cv.LINE_AA)
 

    return result_img


#########################################
#Main 문
path="./Demo_origin.mp4"
cam = cv.VideoCapture(path)
#########################################
from urllib.request import urlopen

# url = "http://172.17.209.76:8080/video" # Report Connect Camera
# cam = cv.VideoCapture(url)

if not cam.isOpened():
    raise RuntimeError("No Video")

check, frame = cam.read()
height, width, channels = frame.shape
height=frame.shape[0]
width=frame.shape[1]

# ROI=np.zeros((height,width,3), np.uint8)
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Final Demo Video.avi', fourcc, 30.0, (width, height))

i=0
while True:
    i=i+1

    check, frame = cam.read()

    frame_processed=Yolo_cnn(frame)

    # 녹화 시작
    out.write(frame_processed)

    cv.namedWindow('frame', flags=cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', width=500, height=400)
    cv.imshow('frame',frame_processed)

    key=cv.waitKey(1)
    if  key == ord("q"):
        break      
    elif key==ord("s"):
        cv.imwrite(f"{i}.jpg",frame)        


# 녹화 종료
cam.release()
out.release()
cv.destroyAllWindows()