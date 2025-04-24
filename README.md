<p align="center">

  <h1 align="center">HybridDyn-VSLAM: Real-Time Visual SLAM for Dynamic Environments using Hybrid Segmentation and Optical Flow</h1>
  <h3 align="center">
    <strong>Yung-Ching Sun</strong>
    ,
    <strong>Peng-Chen Chen</strong>
    ,
    <strong>Chi Zhang</strong>
    ,
    <strong>Hao Yin</strong>
  </h3>
  <h3 align="center"><a href="./doc/poster.pdf">Poster</a> | <a href="./doc/slides.pdf">Slides</a> | <a href="./media/presentation.mp4">Video</a></h3>
  <div align="center"></div>
</p>

1. We propose a dynamic-aware SLAM system that enhances the performance of ORB-SLAM3 in dynamic environments.
2. The system integrates real-time segmentation using YOLO11n and FastSAM to identify potentially dynamic regions in the scene and applies dense optical flow to verify truly dynamic regions.
3. Verified dynamic regions are converted into binary masks to remove dynamic keypoints before tracking, preserving only static features for localization and mapping.
4. Our method is modular, requires no prior scene knowledge, and handles both labeled and unlabeled dynamic objects.
5. Experimental results on the TUM RGB-D and Bonn RGB-D datasets demonstrate significant improvements in localization accuracy and runtime efficiency.

<!-- ## Abstract
In dynamic environments, traditional SLAM systems struggle to maintain accurate localization and mapping due to the presence of moving objects that violate the static-world assumption. To address this challenge, we propose a robust and modular dynamic SLAM framework that enhances ORB-SLAM3 by integrating real-time dynamic region segmentation and optical flow-based motion analysis. Our method leverages FastSAM and YOLO11n-seg to detect potentially dynamic regions, which are further refined using dense optical flow to identify true motion. These dynamic regions are masked to exclude moving region feature points before SLAM processing, enabling improved camera trajectory tracking. Experimental results on the TUM RGB-D and Bonn RGB-D datasets demonstrate significant improvements in localization accuracy and runtime efficiency, achieving real-time performance without requiring prior knowledge of object classes. -->

# Demo
<p align="center">
  <img src="media/tracking_demo.gif" alt="Real-time tracking in dynamic environments" width="720"/>
</p>
<p align="center"><em>Figure 1: The gif shows that our method successfully remove the features of moving object, leading to robust tracking in dynamic environments. </em></p>


# Getting Start
HybridDyn-VSLAM is extended based on [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3).

## ORB-SLAM3 Prerequisites and Environment Setup
To setup the environment for ORB-SLAM3, please follow [ORB-SLAM3 repository](https://github.com/UZ-SLAMLab/ORB_SLAM3). You can also use [ORB-SLAM3 docker](https://github.com/jahaniam/orbslam3_docker).<br> 

<!-- ### 1. Build ORB-SLAM3 docker container
Run
```bash
./build_container_cpu.sh
```
or
```bash
./build_container_cuda.sh
```
depending on your machine.

### 2. Run ORB-SLAM3 docker image
```bash
docker exec -it orbslam3 bash
``` -->

## Masks Generator Environment Setup
```bash
cd Mask_Generator
pip install -r requirements.txt
```

# Examples
### Run RGB-D examples on TUM dataset
1. Download a sequence from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download and uncompress it to Datasets/tum_rgbd/
2. Open the script "HybridDyn_VSLAM/Examples/RGB-D/rgbd_tum_test.sh". Change *pathDatasetTUM* variable to point to the directory where the dataset has been uncompressed.
3. Generate dynamic region masks: 
- Open the script "HybridDyn_VSLAM/Mask_Generator/dy_mask_yolo_fastsam.py". Change *dataset* variable to the directory of the sequence you downloaded. 
- Generate dynamic region masks:
```bash
cd HybridDyn_VSLAM/Mask_Generator
python3 dy_mask_yolo_fastsam.py
```
- Create a mask.txt file to collect the paths of the masks, similar to how it's done in rgb.txt.
- Go to "HybridDyn_VSLAM/Mask_Generator" and run:
```bash
python3 associate.py {PATH_TO_SEQUENCE}/rgb.txt {PATH_TO_SEQUENCE}/depth.txt {PATH_TO_SEQUENCE}/mask.txt > tum_example.txt
```
4. Run the example:
```bash
cd HybridDyn_VSLAM/ORB_SLAM3/Examples/RGB-D
bash ./rgbd_tum_test.sh
```
5. Save the result:
```bash
mkdir results
cp CameraTrajectory.txt ./results/results_test.txt
```
### Run RGB-D examples on Bonn dataset
1. Download a sequence from https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html and uncompress it to Datasets/bonn_rgbd/
2. Open the script "HybridDyn_VSLAM/Examples/RGB-D/rgbd_bonn_test.sh". Change *pathDatasetBonn* variable to point to the directory where the dataset has been uncompressed.
3. Generate dynamic region masks: 
- Open the script "HybridDyn_VSLAM/Mask_Generator/dy_mask_yolo_fastsam.py". Change *dataset* variable to the directory of the sequence you downloaded. 
- Generate dynamic region masks:
```bash
cd HybridDyn_VSLAM/Mask_Generator
python3 dy_mask_yolo_fastsam.py
```
- Create a mask.txt file to collect the paths of the masks, similar to how it's done in rgb.txt.
- Go to "HybridDyn_VSLAM/Mask_Generator" and run:
```bash
python3 associate.py {PATH_TO_SEQUENCE}/rgb.txt {PATH_TO_SEQUENCE}/depth.txt {PATH_TO_SEQUENCE}/mask.txt > tum_example.txt
```
4. Run the example:
```bash
cd HybridDyn_VSLAM/ORB_SLAM3/Examples/RGB-D
bash ./rgbd_bonn_test.sh
```
5. Save the result:
```bash
mkdir results
cp CameraTrajectory.txt ./results/results_test.txt
```
# Evaluation
For TUM datset examples: Open the script "HybridDyn_VSLAM/evaluation/eval_tum_ate.sh". Change *pathDatasetTum* variable to point to the directory of your sequence. Run:
```bash
cd HybridDyn_VSLAM/ORB_SLAM3/evaluation
mkdir plots
bash ./eval_tum_ate.sh
```
For Bonn dataset examples: Open the script "HybridDyn_VSLAM/evaluation/eval_bonn_ate.sh". Change *pathDatasetBonn* variable to point to the directory of your sequence. Run:
```bash
cd HybridDyn_VSLAM/ORB_SLAM3/evaluation
mkdir plots
bash ./eval_bonn_ate.sh
```
<!-- # Demo
<p align="center">
  <img src="media/tracking_demo.gif" alt="Real-time tracking in dynamic environments" width="720"/>
</p>
<p align="center"><em>Figure 1: The gif shows that our method successfully remove the features of moving object, leading to robust tracking in dynamic environments. </em></p>  -->

# Acknowledgement
We would like to thank Prof. Maani Ghaffari, Dr. Minghan Zhu, and the GSIs for their inspiring lectures, helpful instruction, and support during the semester and throughout this project. 

# References
[1] C. Campos, R. Elvira, J. J. G´omez Rodr´ıguez, J. M. M. Mon-
tiel, and J. D. Tard´os, “ORB-SLAM3: An Accurate Open-Source
Library for Visual, Visual–Inertial, and Multimap SLAM,” IEEE
Transactions on Robotics, vol. 37, no. 6, pp. 1874–1890, 2021. ORB-SLAM3 Github Repository: https://github.com/UZ-SLAMLab/ORB_SLAM3<br>
[2] B. Bescos, J. M. F´acil, J. Civera, and J. Neira,
“DynaSLAM: Tracking, mapping, and inpainting in
dynamic scenes,” IEEE robotics and automation
letters, vol. 3, no. 4, pp. 4076–4083, 2018. DynaSLAM Github Repository: https://github.com/BertaBescos/DynaSLAM<br>
[3] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and
D. Cremers, “A benchmark for the evaluation of
rgb-d slam systems,” in Proc. of the International
Conference on Intelligent Robot Systems (IROS),
Oct. 2012. TUM Dataset available at: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download<br>
[4] E. Palazzolo, J. Behley, P. Lottes, P. Gigu`ere, and
C. Stachniss, “ReFusion: 3D Reconstruction in Dy-
namic Environments for RGB-D Cameras Exploit-
ing Residuals,” 2019. Bonn RGB-D Dataset availabe at: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html<br>
[5] ORB-SLAM3 docker with GUI Github repository: https://github.com/jahaniam/orbslam3_docker
