#!/bin/bash
pathDatasetBonn='/Datasets/bonn_rgbd' #Example, it is necesary to change it by the dataset path

./rgbd_tum ../../Vocabulary/ORBvoc.txt ./TUM3.yaml "$pathDatasetBonn"/rgbd_bonn_moving_nonobstructing_box "$pathDatasetBonn"/rgbd_bonn_moving_nonobstructing_box/moving_no_box.txt
