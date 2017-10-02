#!/usr/bin/env python

import numpy as np
import os

import chainer
from chainercv.utils import read_image

def get_class_list(class_path):
    list = []
    for line in open(class_path, 'r'):
        line = line.rstrip("\n")
        pair = line.split()
        list.append(pair[0]) # name
    return list

class DetectionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_directory, label_names):
        self.data_dir = data_directory
        self.label_names = label_names

        self.img_filenames = sorted(os.listdir(self.data_dir + "/images/"))
        self.anno_filenames = sorted(os.listdir(self.data_dir + "/labels/"))

    def __len__(self):
        return len(self.img_filenames)

    def get_example(self, i):
        img_filename = self.data_dir + "/images/" + self.img_filenames[i]
        anno_filename = self.data_dir + "/labels/" + self.anno_filenames[i]
        img = read_image(img_filename)

        bboxes = []
        labels = []
        for line in open(anno_filename, "r"):
            line = line.rstrip("\n")
            label = line.split()[0]
            rect_str = line.split()[4:8]
            rect = []
            for i in xrange(len(rect_str)):
                rect.append(int(float(rect_str[i])))
            bboxes.append([rect[1], rect[0], rect[3], rect[2]]) # y_min, x_min, y_max, x_max
            labels.append(self.label_names.index(label))

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return img, bboxes, labels
