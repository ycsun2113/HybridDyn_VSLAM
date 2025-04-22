#!/bin/bash
pathDatasetBonn='/Datasets/bonn_rgbd' #Example, it is necesary to change it by the dataset path

python3 evaluate_ate_tum.py "$pathDatasetBonn"/rgbd_bonn_moving_nonobstructing_box/groundtruth.txt ../Examples/RGB-D/results/mov_box.txt --plot ./plots/mov_box.pdf --verbose
