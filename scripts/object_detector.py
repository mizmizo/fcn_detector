#!/usr/bin/env python

import rospy
import caffe
import os
import sys
import math
import numpy as np
import random
import cv2 as cv
import matplotlib.pylab as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from fcn_detector.msg import Box, BoxArray

class FCNObjectDetector():

    def __init__(self):
        self.__net = None
        self.__transformer = None
        self.__im_width = None
        self.__im_height = None
        self.__bridge = CvBridge()
        self.__labels = {}

        self.__prob_thresh = rospy.get_param('~detection_threshold', 0.5)  #! threshold for masking the detection
        self.__min_bbox_thresh = rospy.get_param('~min_boxes', 3) #! minimum bounding box
        self.__group_eps_thresh = rospy.get_param('~nms_eps', 0.2) #! bbox grouping
        self.__weights = rospy.get_param('~pretrained_weights', None)
        self.__model_proto = rospy.get_param('~deployment_prototxt', None)
        self.__device_id = rospy.get_param('device_id', 0)
        self.__class_list = rospy.get_param('~class_list', None) #! path to class list
        self.__visualize = rospy.get_param('~visualize', True)

        if self.is_file_valid():
            self.load_caffe_model()
            rospy.loginfo('DETECTOR SETUP SUCCESSFUL')

        ##! publisher setup
        self.pub_box = rospy.Publisher('~boxes', BoxArray, queue_size = 1)
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

        random.seed(5) #! same seed with image_annotation tool
        input_image = cv_img.copy()

        caffe.set_device(self.__device_id)
        caffe.set_mode_gpu()

        self.__net.blobs['data'].data[...] = self.__transformer.preprocess('data', cv_img)
        output = self.__net.forward()

        object_boxes = []
        label_color = []
        object_colors = []
        object_labels = []
        rand_list = [random.randint(0, 255) for i in xrange(len(self.__labels) * 3)]
        for i in xrange(len(self.__labels)):
            label_color.append(rand_list[i * 3:(i + 1) * 3])

        for index, key in enumerate(sorted(output.keys())):
            bboxes = output[key]
            obj_boxes = bboxes[(np.nonzero(np.dot(bboxes, np.array([1.,1.,1.,1.,1.]))))]
            if obj_boxes.any():
                label = self.__labels[str(index + 1)]
                for box in obj_boxes:
                    object_boxes.append(box)
                    object_labels.append(label)
                    object_colors.append(label_color[index + 1])

        label_color = np.asarray(label_color, dtype=np.float)
        object_boxes = np.asarray(object_boxes, dtype=np.int)

        if not len(object_boxes):
            rospy.logwarn("not detection")
            return
        else:
            print "found objects : %s" % object_labels

        cv_img = cv.resize(input_image.copy(), (input_image.shape[1], input_image.shape[0]))
        im_out = cv_img.copy()
        [
            [
                cv.rectangle(cv_img, (box[0], box[1]), (box[2], box[3]), object_colors[index], -1),
                cv.rectangle(cv_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
            ] for index, box in enumerate(object_boxes)
        ]

        ##! publish and visualize
        boxes = BoxArray()
        for obj_box, obj_label in zip(object_boxes, object_labels):
            box = Box()
            box.label = obj_label
            box.x = obj_box[0]
            box.y = obj_box[1]
            box.width = obj_box[0] - obj_box[2]
            box.height = obj_box[1] - obj_box[3]
            boxes.boxes.append(box)

        boxes.header = image_msg.header

        alpha = 0.3
        cv.addWeighted(im_out, alpha, cv_img, 1.0 - alpha, 0, im_out)

        imout_msg = self.__bridge.cv2_to_imgmsg(im_out, "bgr8")
        imout_msg.header = image_msg.header

        if self.__visualize:
            cv.namedWindow('detection', cv.WINDOW_NORMAL)
            cv.imshow('detection', im_out)
            cv.waitKey(3)

        self.pub_box.publish(boxes)
        self.pub_img.publish(imout_msg)


    def callback(self, image_msg):
        self.run_detector(image_msg)


    def load_caffe_model(self):
        rospy.loginfo('LOADING CAFFE MODEL..')
        self.__net = caffe.Net(self.__model_proto, self.__weights, caffe.TEST)

        self.__transformer = caffe.io.Transformer({'data': self.__net.blobs['data'].data.shape})
        self.__transformer.set_transpose('data', (2,0,1))
        self.__transformer.set_raw_scale('data', 1)
        self.__transformer.set_channel_swap('data', (2,1,0))

        shape = self.__net.blobs['data'].data.shape
        self.__im_height = shape[2]
        self.__im_width = shape[3]

        self.__net.blobs['data'].reshape(1, 3, self.__im_height, self.__im_width)

        for line in open(self.__class_list, 'r'):
            line = line.rstrip("\n")
            pair = line.split()
            self.__labels[pair[1]] = pair[0]


    def subscribe(self):
        rospy.Subscriber('image', Image, self.callback, tcp_nodelay=True)


    def is_file_valid(self):
        if self.__model_proto is None or \
           self.__weights is None or \
           self.__class_list is None:
            rospy.logfatal('PROVIDE PRETRAINED MODEL! KILLING NODE...')
            return False

        is_file = lambda path : os.path.isfile(str(path))
        if  (not is_file(self.__model_proto)) or (not is_file(self.__weights)) or (not is_file(self.__class_list)):
            rospy.logfatal('NOT SUCH FILES')
            return False

        return True


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
