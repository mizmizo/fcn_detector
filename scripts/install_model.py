#!/usr/bin/env python

import argparse
import multiprocessing
import os.path as osp

import jsk_data

def download_data(*args, **kwargs):
    p = multiprocessing.Process(
            target=jsk_data.download_data,
            args=args,
            kwargs=kwargs)
    p.start()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    quiet = args.quiet

    PKG = 'fcn_detector'

    download_data(
        pkg_name=PKG,
        path='models/chainer/whole_class_detector/model.npz',
        url='https://drive.google.com/uc?id=0B09VRnpQxd6PdnphYVZScjkwa1U',
        md5='e7f47bbf8ad72fbd7b329e00a5290405',
        quiet=quiet,
    )

    download_data(
        pkg_name=PKG,
        path='models/chainer/whole_class_detector/class_list.txt',
        url='https://drive.google.com/uc?id=0B09VRnpQxd6PYmQwM1lPMGpVQlE',
        md5='b365e4885a2a59b5bd4e926bdfd7b1cb',
        quiet=quiet,
    )

if __name__ == '__main__':
    main()


