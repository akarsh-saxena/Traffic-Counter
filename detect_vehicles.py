# pip install yolo34py-gpu

from pydarknet import Detector, Image
import cv2
import numpy as np
import pickle as pkl

def detect(cfg, weights, coco_data, source_video):

    print('Detecting vehicles (This may take time)...')

    net = Detector(bytes(cfg, encoding="utf-8"), bytes(weights, encoding="utf-8"), 0, bytes(coco_data,encoding="utf-8"))

    detections = []

    i = 0
    cap = cv2.VideoCapture(source_video)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        img = Image(frame)
        results = net.detect(img)

        if i%500==0:
            print(i)

        i+=1
        
        detections.append(results)

    pkl.dump(detections, open('detections.pkl', 'wb'))
