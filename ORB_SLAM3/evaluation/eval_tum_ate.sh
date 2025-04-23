#!/bin/bash
pathDatasetTUM='/Datasets/tum_rgbd' #Example, it is necesary to change it by the dataset path

python3 evaluate_ate_tum.py "$pathDatasetTUM"/rgbd_dataset_freiburg3_walking_static/groundtruth.txt ../Examples/RGB-D/results/fr3_w_static.txt --plot ./plots/fr3_w_static.pdf --verbose

