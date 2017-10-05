#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plot
import yaml

import rospy
import os
import sys
import math
import numpy as np
import random
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from fcn_detector.msg import ScoredBox, ScoredBoxArray

import chainer
from chainercv.datasets import voc_detection_label_names
from chainercv.links import SSD512
from train_utils import train
from chainercv import utils
from chainercv.visualizations import vis_bbox

from detection_dataset import DetectionDataset, get_class_list

class FCNObjectDetector():

    def __init__(self):
        self.__model = None
        self.__bridge = CvBridge()
        self.__labels = {}
        self.__label_color = []

        self.__gpu = rospy.get_param('~gpu', 0)  #! threshold for masking the detection
        self.__pretrained_model = rospy.get_param('~pretrained_model', None)
        self.__class_list = rospy.get_param('~class_list', None) #! path to class list
        self.__score_thresh = rospy.get_param('~score_thresh', 0.6)
        self.__visualize = rospy.get_param('~visualize', False)

        if self.is_file_valid():
            self.load_model()
            rospy.loginfo('DETECTOR SETUP SUCCESSFUL')
            rospy.loginfo('Detect thresh: %s' % self.__model.score_thresh)

        random.seed(5) #! same seed with image_annotation tool
        rand_list = [random.randint(0, 255) for i in xrange(len(self.__labels) * 3)]
        for i in xrange(len(self.__labels)):
            self.__label_color.append(rand_list[i * 3:(i + 1) * 3])
        self.__label_color = np.asarray(self.__label_color, dtype=np.float)

        ##! publisher setup
        self.pub_box = rospy.Publisher('~boxes', ScoredBoxArray, queue_size = 1)
        self.pub_img = rospy.Publisher('~image', Image, queue_size = 1)
        self.subscribe()

    def run_detector(self, image_msg):
        cv_img = None
        try:
            cv_img = self.__bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception as e:
            print (e)
            return
        if cv_img is None:
            return

        img = cv_img.copy()
        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            chainer_img = img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            chainer_img =  img.transpose((2, 0, 1))

        _bboxes, _labels, _scores = self.__model.predict([chainer_img])
        bboxes, labels, scores = _bboxes[0], _labels[0], _scores[0]

        if not len(bboxes):
            rospy.logwarn("not detection")
            return
        else:
            rospy.loginfo("found objects : %s" % [ self.__labels[label] for label in labels ])

        ##! publish and visualize
        boxes = ScoredBoxArray()
        for bbox, label, score in zip(bboxes, labels, scores):
            box = ScoredBox()
            box.label = self.__labels[label]
            box.x = bbox[1]
            box.y = bbox[0]
            box.width = bbox[3] - bbox[1]
            box.height = bbox[2] - bbox[0]
            box.score = score
            boxes.boxes.append(box)

        im_out = self.draw_label(img, bboxes, labels, scores)

        boxes.header = image_msg.header
        imout_msg = self.__bridge.cv2_to_imgmsg(im_out, "bgr8")
        imout_msg.header = image_msg.header

        if self.__visualize:
            cv.namedWindow('detection', cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL)
            cv.imshow('detection', im_out)
            cv.waitKey(3)
        self.pub_box.publish(boxes)
        self.pub_img.publish(imout_msg)


    def load_model(self):
        self.__labels = get_class_list(self.__class_list)
        self.__model = SSD512(
            n_fg_class=len(self.__labels),
            pretrained_model=self.__pretrained_model)
        if self.__gpu >= 0:
            chainer.cuda.get_device_from_id(self.__gpu).use()
            self.__model.to_gpu()
        self.__model.score_thresh = self.__score_thresh


    def is_file_valid(self):
        if self.__pretrained_model is None or \
           self.__class_list is None:
            rospy.logfatal('PROVIDE PRETRAINED MODEL! KILLING NODE...')
            return False

        is_file = lambda path : os.path.isfile(str(path))
        if  (not is_file(self.__pretrained_model)) or (not is_file(self.__class_list)):
            rospy.logfatal('NOT SUCH FILES')
            return False

        return True


    def draw_label(self, img, bboxes, labels, scores):
        rect_img = cv.resize(img.copy(), (img.shape[1], img.shape[0]))
        im_out = rect_img.copy()
        [
            [
                cv.rectangle(rect_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), self.__label_color[label], -1),
                #cv.rectangle(rect_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), self.__label_color[label], 4),
                cv.rectangle(rect_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 4),
                cv.putText(rect_img, self.__labels[label] + ": " + "{0:.2f}".format(score), (int(bbox[1]), int(bbox[0]) - 5), cv.FONT_HERSHEY_TRIPLEX, 1.0,self.__label_color[label])
            ] for bbox, label, score in zip(bboxes, labels, scores)
        ]
        alpha = 0.4
        cv.addWeighted(im_out, alpha, rect_img, 1.0 - alpha, 0, im_out)
        return im_out


    def callback(self, image_msg):
        self.run_detector(image_msg)


    def subscribe(self):
        rospy.Subscriber('image', Image, self.callback, tcp_nodelay=True)


def main(argv):
    try:
        rospy.init_node('fcn_object_detector', anonymous = True)
        fod = FCNObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass


if __name__ == "__main__":
    main(sys.argv)
