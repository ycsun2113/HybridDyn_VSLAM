from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import requests
import os
from tqdm import tqdm

# Change the path to your datasets' path
# datasets_path = "../orbslam3_docker/Datasets/tum_rgbd/"
# dataset = "rgbd_dataset_freiburg3_walking_halfsphere/"
datasets_path = "../orbslam3_docker/Datasets/bonn_rgbd/"
dataset = "rgbd_bonn_moving_nonobstructing_box/"
input_dir = datasets_path + dataset + "rgb/"
image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")])

output_dir = datasets_path + dataset + "mask_fastsam/"
os.makedirs(output_dir, exist_ok=True)

if torch.cuda.is_available():
    model_yolo = YOLO("yolo11n-seg.pt").cuda()
    model_fastsam = FastSAM("FastSAM-s.pt").cuda()
else:
    model_yolo = YOLO("yolo11n-seg.pt")
    model_fastsam = FastSAM("FastSAM-s.pt")

# Try different threshold for your need
flow_diff_thres = 1.7
yolo_flow_thres = 1.15

visualize = False
dynamic_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

for i in tqdm(range(len(image_paths) - 1)):
    img1 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths[i+1])

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Step 1-1: Segmentation with yolo11n-seg
    if torch.cuda.is_available():
        results_yolo = model_yolo(img1, device=0, verbose=False)
    else:
        results_yolo = model_yolo(img1, device="cpu", verbose=False)
    result_yolo = results_yolo[0]

    if result_yolo.masks is None:
        # print(f"No objects detected in frame {i}. Treating as no mask.")
        masks_yolo = np.zeros((1, img1.shape[0], img1.shape[1]), dtype=bool)  
    else:
        masks_yolo = result_yolo.masks.data.cpu().numpy()

    # Step 1-2: Segmentation with FastSAM
    if torch.cuda.is_available():
        results_fastsam = model_fastsam(img1, device=0, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
    else:
        results_fastsam = model_fastsam(img1, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
    result_fastsam = results_fastsam[0]
    masks_fastsam = result_fastsam.masks.data.cpu().numpy()

    # Step 2: Compute Optical Flow
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    magnitude = np.linalg.norm(flow, axis=2)

    # Step 3: Calculate background's avgerage optical flow 
    all_masks_yolo = np.sum(masks_yolo, axis=0) > 0 # masked areas
    background_mask_yolo = ~all_masks_yolo
    background_flow_vals_yolo = magnitude[background_mask_yolo]
    background_flow_mean_yolo = background_flow_vals_yolo.mean()
    # print(f"Background flow mean: {background_flow_mean:.2f}")

    all_masks_fastsam = np.sum(masks_yolo, axis=0) > 0 # masked areas
    background_mask_fastsam = ~all_masks_fastsam
    background_flow_vals_fastsam = magnitude[background_mask_fastsam]
    background_flow_mean_fastsam = background_flow_vals_fastsam.mean()

    # Step 4: Check which mask is dynamic region
    overlay = img1_rgb.copy()
    dynamic_objects_mask = []

    # Using FastSAM
    for idx, mask in enumerate(masks_fastsam):

        # compute optical flow for masking
        mask_bool = mask.astype(bool)
        flow_vals = magnitude[mask_bool]
        avg_flow = flow_vals.mean()

        if background_flow_mean_yolo > 1.0:  # if background optical flow is relatively large, use rate
            flow_diff = avg_flow / background_flow_mean_yolo
        else:  # if background optical flow is relatively small, use flow difference
            flow_diff = abs(avg_flow - background_flow_mean_yolo)

        # Step 5: Mask highly dynamic objects
        if idx != 0:
            class_id = int(result_fastsam.boxes.cls[idx])
            class_name = result_fastsam.names[class_id]
            is_dynamic = (class_name in dynamic_classes) or (flow_diff > flow_diff_thres)
        else:
            is_dynamic = (flow_diff > flow_diff_thres)
            
        if is_dynamic:
            dynamic_objects_mask.append(mask)

    # Step 6: Create mask for ORB-SLAM3
    orb_slam_mask_input = np.zeros_like(img1[:, :, 0], dtype=np.uint8)
    kernel = np.ones((9, 9), np.uint8)

    for dynamic_mask in dynamic_objects_mask:
        orb_slam_mask_input[dynamic_mask.astype(bool)] = 255

    orb_slam_mask_input = cv2.dilate(orb_slam_mask_input, kernel, iterations=1)

    # Step 7: Save binary masks
    input_filename = os.path.basename(image_paths[i])
    output_path = os.path.join(output_dir, input_filename)

    cv2.imwrite(output_path, orb_slam_mask_input)
    # print(f"Saved masked image: {output_path}")

input_filename = os.path.basename(image_paths[-1])
output_path = os.path.join(output_dir, input_filename)

last_img = cv2.imread(image_paths[-1])
last_img = np.zeros_like(last_img[:, :, 0], dtype=np.uint8)
cv2.imwrite(output_path, last_img)