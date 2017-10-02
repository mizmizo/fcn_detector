#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plot
import yaml

import chainer

from chainercv.datasets import voc_detection_label_names
from chainercv.links import SSD512
from train_utils import train
from chainercv import utils
from chainercv.visualizations import vis_bbox

from detection_dataset import DetectionDataset, get_class_list

def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model')
    parser.add_argument(
        '--class_list', help='The path to the txt file with label names')
    parser.add_argument('image')
    args = parser.parse_args()

    label_names = get_class_list(args.class_list)
    model = SSD512(
        n_fg_class=len(label_names),
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]
    # bbox : [y0, x0, y1, x0]

    vis_bbox(
        img, bbox, label, score, label_names=label_names)
    plot.show()

if __name__ == '__main__':
    main()
