import pickle as pkl
import math
import numpy as np
import cv2
import detect_vehicles

import argparse

def count(path):

    print('Counting Cars...')

    detections = pkl.load(open('detections.pkl', 'rb'))

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    total_cars = 0
    total_trucks = 0

    line_coordinates = [(50, 350), (1230, 350)]

    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)

    while True:

        ret, img = cap.read()

        if not ret:
            break

        found = False

        for cat, score, bounds in detections[i]:
            x, y, w, h = bounds
            if int(math.fabs(y-int(line_coordinates[0][1]))) <= 8 and y < int(line_coordinates[0][1]):
                color = green
                found = True
                if str(cat.decode('utf-8')) == 'car':
                    total_cars += 1
                elif str(cat.decode('utf-8')) == 'truck':
                    total_trucks += 1
            else:
                color = red
            cv2.circle(img, (int(x), int(y)), 2, color, 2)
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, thickness=2)
            if not found:
                cv2.line(img, line_coordinates[0], line_coordinates[1], color, thickness=5)
            else:
                cv2.line(img, line_coordinates[0], line_coordinates[1], green, thickness=5)
            cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX,1,color)
            cv2.putText(img, 'Total Cars: {}'.format(total_cars), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            cv2.putText(img, 'Total Trucks: {}'.format(total_trucks), (1000, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                print('Force closing window...')
                cap.release()
                cv2.destroyAllWindows()
        i+=1

        cv2.imshow('', img)
        cv2.waitKey(int( (1 / int(29)) * 1000))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count Vehicles',
                                     usage='run -s [source-video-path]')
    parser.add_argument('-s', '--source-video', help='Source video to count vehicles', action='store', required=True)
    parser.add_argument('-w', '--yolo-weights', help='Path to yolo weights file', action='store', required=True)
    parser.add_argument('-c', '--yolo-config', help='Path to yolo config file', action='store', required=True)
    parser.add_argument('-cd', '--coco-data', help='Path to coco data file', action='store', required=True)
    args = parser.parse_args()

    source_video = vars(args)['source_video']
    weights = vars(args)['yolo_weights']
    config = vars(args)['yolo_config']
    coco_data = vars(args)['coco_data']

    detect_vehicles.detect(config, weights, coco_data, source_video)
    count(source_video)
