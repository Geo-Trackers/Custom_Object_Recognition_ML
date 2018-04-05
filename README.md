# Deep Learning - Custom Aerial Image Recognition 

This repository contains a set of python scripts for creating a list of annotations(in xml) based on the user defined bounding box on multiple set of aerial images and the visualization of the detection accuracy. This list of annotation files can thus be used for the training purpose using Deep-Neural-Networks (YOLO v.2 neural detectors) based on Darknet( A custom GPU-accelerated Framework) for near real time aerial object detection and classification. After labelling, the set of images with their accurate bounding box were trained on GPU accelerated server with NVIDIA Tesla P100 SMI 16 GB for 5 hours until 4000 epochs. The resulting weights and configuration files were then loaded on the script (visualization.py) to visualize the detection accuracy.

Dependencies: 

1. OpenCV 3
2. Python 3, Matplotlib, Numpy
3. Darkflow(Open Source Neural Network Darknet in C translated to tensorflow) 
4. CUDA 
5. Tensorflow 1.0 

Results

<img> 
![unnamed](https://user-images.githubusercontent.com/7304644/38349477-7f320758-38d1-11e8-966a-dd59ca29aa1f.png
</img>



ENJOY
Thank you 
