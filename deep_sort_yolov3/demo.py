#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque, Counter
from keras import backend
import torch
import torchvision.transforms as T
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw

values = ['Age1-16', 'Age17-30', 'Age31-45', 'Age46-60', 'Female', 'Male']

backend.clear_session()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="/home/deep/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/videos/abc.mp4")
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
        '/home/deep/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/exp_result/custom/custom/img_model/ckpt_max.pth')
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


def main(yolo):
    start = time.time()
    # Definition of the parameters
    max_cosine_distance = 0.5  # 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3  # 非极大抑制的阈值

    counter = []
    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    # video_path = "./output/output.avi"
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/' + args["input"][43:57] + "_" + args["class"] + '_output.avi', fourcc, 15,
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

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        attribute_dict = {"Male" : 0, "Female" : 0, 'Age1-16' : 0, 'Age17-30' : 0, 'Age31-45' : 0, 'Age46-60' : 0}
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            crop_img = image.crop([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            attributes = age_group = gender = None
            if track.hits <= 100:
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
                if track.track_id not in hourly_people_counter:
                    hourly_attribute_dict[age_group[0]] += 1
                    hourly_attribute_dict[gender[0]] += 1
                    hourly_people_counter.add(int(track.track_id))
                if track.track_id not in daily_people_counter:
                    daily_attribute_dict[age_group[0]] += 1
                    daily_attribute_dict[gender[0]] += 1
                    daily_people_counter.add(int(track.track_id))

            i += 1

        count = len(set(counter))
        current_time = time.time()

        # if int((current_time - hourly_timer)) > 1:
        realtime_file = open("current.txt", "a")
        realtime_file.write(
            "Total People:  " + str(count) +
            "\tMale:  " + str(attribute_dict["Male"]) +
            "\tFemale:  " + str(attribute_dict["Female"]) +
            "\tAge1-16:  " + str(attribute_dict["Age1-16"]) +
            "\tAge17-30:  " + str(attribute_dict["Age17-30"]) +
            "\tAge31-45:  " + str(attribute_dict["Age31-45"]) +
            "\tAge46-60:  " + str(attribute_dict["Age46-60"]) + "\n")
        realtime_file.close()
        if int((current_time - hourly_timer)/3600) >= 1:
            hourly_timer = current_time
            hourly_file = open("hourly.txt", "a")
            hourly_file.write(
                "Start Hour:" + (datetime.datetime.now() - datetime.timedelta(hours = 1)).strftime("%d/%m/%Y %H:%M:%S") +
                "\tEnd Hour:" + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                "\tTotal People:  " + str(count) +
                "\tMale:  " + str(hourly_attribute_dict["Male"]) +
                "\tFemale:  " + str(hourly_attribute_dict["Female"]) +
                "\tAge1-16:  " + str(hourly_attribute_dict["Age1-16"]) +
                "\tAge17-30:  " + str(hourly_attribute_dict["Age17-30"]) +
                "\tAge31-45:  " + str(hourly_attribute_dict["Age31-45"]) +
                "\tAge46-60:  " + str(hourly_attribute_dict["Age46-60"]) + "\n")
            hourly_file.close()
            hourly_people_counter = set()
            hourly_attribute_dict = {"Male": 0, "Female": 0, 'Age1-16': 0, 'Age17-30': 0, 'Age31-45': 0, 'Age46-60': 0}

        if int((current_time - daily_timer) / 3600) >= 24:
            daily_timer = current_time
            daily_file = open("daily.txt", "a")
            daily_file.write(
                "Date: " + str(datetime.date.today()) +
                "\tTotal People:  " + str(count) +
                "\tMale:  " + str(hourly_attribute_dict["Male"]) +
                "\tFemale:  " + str(hourly_attribute_dict["Female"]) +
                "\tAge1-16:  " + str(hourly_attribute_dict["Age1-16"]) +
                "\tAge17-30:  " + str(hourly_attribute_dict["Age17-30"]) +
                "\tAge31-45:  " + str(hourly_attribute_dict["Age31-45"]) +
                "\tAge46-60:  " + str(hourly_attribute_dict["Age46-60"]) + "\n")
            daily_file.close()
            daily_people_counter = set()
            daily_attribute_dict = {"Male": 0, "Female": 0, 'Age1-16': 0, 'Age17-30': 0, 'Age31-45': 0, 'Age46-60': 0}

        cv2.putText(frame, "Total Object Counter: " + str(count), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "Current Object Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO3_Deep_SORT', frame)

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
        # print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    # if len(pts[track.track_id]) != None:
    #     print(args["input"][43:57] + ": " + str(count) + " " + str(class_name) + ' Found')
    #
    # else:
    #     print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
