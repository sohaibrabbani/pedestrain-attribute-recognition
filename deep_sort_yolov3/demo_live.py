#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import argparse
import datetime
import time
import warnings
from collections import deque, Counter
import cv2
import imutils
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, Response, render_template
from imutils.video import VideoStream
from keras import backend

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools import generate_detections as gdet
from yolo import YOLO

values = ['Age1-16', 'Age17-30', 'Age31-45', 'Age46-60', 'Female', 'Male']

backend.clear_session()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="/home/deep/PycharmProjects/pedestrian-attribute-recognition/videos/abc.mp4")
ap.add_argument("-c", "--class", help="name of class", default="person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")


def model_init_par():
    # model
    backbone = resnet50()
    classifier = BaseClassifier(nattr=6)
    model = FeatClassifier(backbone, classifier)

    # load
    checkpoint = torch.load(
       '/home/deep/PycharmProjects/pedestrain-attribute-recognition/exp_result/custom/custom/img_model/ckpt_max.pth')
    # unfolded load
    # state_dict = checkpoint['state_dicts']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # one-liner load
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()
    #     model.load_state_dict(checkpoint['state_dicts'])
    # else:
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dicts'].items()})
    # cuda eval
    model.cuda()
    model.eval()

    # valid_transform
    height, width = 256, 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    return model, valid_transform


def demo_par(model, valid_transform, img):
    # load one image
    img_trans = valid_transform(img)
    imgs = torch.unsqueeze(img_trans, dim=0)
    imgs = imgs.cuda()
    valid_logits = model(imgs)
    valid_probs = torch.sigmoid(valid_logits)
    score = valid_probs.data.cpu().numpy()

    # show the score in the image
    txt_res = []
    age_group = []
    gender = []
    txt = ""
    for idx in range(len(values)):
        if score[0, idx] >= 0.5:
            temp = '%s: %.2f ' % (values[idx], score[0, idx])
            if idx < 4:
                age_group.append(values[idx])
            else:
                gender.append(values[idx])
            # txt += temp
            txt_res.append(temp)
    return txt_res, age_group, gender
yolo = YOLO()
max_cosine_distance = 0.5  # 余弦距离的控制阈值
nn_budget = None
model_filename = 'model_data/market1501.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


def get_tracking_info(frame):
    nms_max_overlap = 0.3
    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
    boxs, class_names = yolo.detect_image(image)
    features = encoder(frame, boxs)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)

    return image, boxs


def main():
    start = time.time()
    # Definition of the parameters
    global yolo, encoder, metric, tracker, nn_budget, max_cosine_distance

    counter = []
    # deep_sort

    writeVideo_flag = True
    # video_path = "./output/output.avi"
    # video_capture = cv2.VideoCapture(args["input"])
    ip_camera = 'rtsp://admin:cvml@123@10.11.18.114'
    camera = 'rtsp://kics.uet:kics@12345@10.11.7.82:554/cam/realmonitor?channel=1&subtype=0'
    video_capture = cv2.VideoCapture(ip_camera)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('/home/deep/PycharmProjects/pedestrian-attribute-recognition/output/' + "_" + '_output.avi', fourcc, 15,
                              (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    model_par, valid_transform = model_init_par()

    hourly_timer = time.time()
    daily_timer = time.time()
    hourly_attribute_dict = {"Male": 0, "Female": 0, 'Age1-16': 0, 'Age17-30': 0, 'Age31-45': 0, 'Age46-60': 0}
    daily_attribute_dict = {"Male": 0, "Female": 0, 'Age1-16': 0, 'Age17-30': 0, 'Age31-45': 0, 'Age46-60': 0}
    daily_people_counter = set()
    hourly_people_counter = set()
    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        image, boxs = get_tracking_info(frame)

        i = int(0)
        indexIDs = []
        attribute_dict = {"Male" : 0, "Female" : 0, 'Age1-16' : 0, 'Age17-30' : 0, 'Age31-45' : 0, 'Age46-60' : 0}
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            crop_img = image.crop([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            attributes = age_group = gender = None
            if track.hits <= 30:
                attributes, age_group, gender = demo_par(model_par, valid_transform, crop_img)
                track.age_group_list.extend(age_group)
                track.gender_list.extend(gender)
            elif not track.attributes:
                track.attributes.append(Counter(track.age_group_list).most_common(1)[0][0])
                track.attributes.append(Counter(track.gender_list).most_common(1)[0][0])
                attributes = track.attributes
            else:
                attributes = track.attributes

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(attributes), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, color, 2)

            if not age_group and not gender:
                age_group = Counter(track.age_group_list).most_common(1)[0]
                gender = Counter(track.gender_list).most_common(1)[0]

            if gender and age_group:
                attribute_dict[gender[0]] += 1
                attribute_dict[age_group[0]] += 1

                hourly_attribute_dict, hourly_people_counter = update_attribute_count(track.track_id, hourly_people_counter, hourly_attribute_dict, age_group[0], gender[0])
                daily_attribute_dict, daily_people_counter = update_attribute_count(track.track_id, daily_people_counter, daily_attribute_dict, age_group[0], gender[0])

            i += 1
        cv2.putText(frame, "CURRENT - " + str(attribute_dict)[1:-1], (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, "HOURLY - " + str(hourly_attribute_dict)[1:-1], (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _frame = imutils.resize(frame, width=800)
        (flag, encodedImage) = cv2.imencode(".jpg", _frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

        count = len(set(counter))
        current_time = time.time()

        realtime_file = open("current.txt", "a")
        realtime_file.write(get_age_gender_text(count, attribute_dict))
        realtime_file.close()

        if int((current_time - hourly_timer)/3600) >= 1:
            hourly_timer = current_time
            hourly_file = open("hourly.txt", "a")
            hourly_file.write(
                "Start Hour:" + (datetime.datetime.now() - datetime.timedelta(hours = 1)).strftime("%d/%m/%Y %H:%M:%S") +
                "\tEnd Hour:" + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                "\t" + get_age_gender_text(count, hourly_attribute_dict))
            hourly_file.close()
            hourly_people_counter = set()
            hourly_attribute_dict = {"Male": 0, "Female": 0, 'Age1-16': 0, 'Age17-30': 0, 'Age31-45': 0, 'Age46-60': 0}

        if int((current_time - daily_timer) / 3600) >= 24:
            daily_timer = current_time
            daily_file = open("daily.txt", "a")
            daily_file.write(
                "Date: " + str(datetime.date.today()) +
                "\t" + get_age_gender_text(count, daily_attribute_dict))
            daily_file.close()
            daily_people_counter = set()
            daily_attribute_dict = {"Male": 0, "Female": 0, 'Age1-16': 0, 'Age17-30': 0, 'Age31-45': 0, 'Age46-60': 0}

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = (fps + (1. / (time.time() - t1))) / 2
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def update_attribute_count(id, counter, attr_dict, age, gender):
    if id not in counter:
        attr_dict[age] += 1
        attr_dict[gender] += 1
        counter.add(int(id))
    return attr_dict, counter

def get_age_gender_text(total_count, attribute_dict):
    return  (
            "Total People:  " + str(total_count) +
            "\tMale:  " + str(attribute_dict["Male"]) +
            "\tFemale:  " + str(attribute_dict["Female"]) +
            "\tAge1-16:  " + str(attribute_dict["Age1-16"]) +
            "\tAge17-30:  " + str(attribute_dict["Age17-30"]) +
            "\tAge31-45:  " + str(attribute_dict["Age31-45"]) +
            "\tAge46-60:  " + str(attribute_dict["Age46-60"]) + "\n")

outputFrame = None
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(main(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run()