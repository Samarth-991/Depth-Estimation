# Depth-Estimation
Monocular Depth Estimation is the task of estimating the depth value (distance relative to the camera) of each pixel given a single (monocular) RGB image. This challenging task is a key prerequisite for determining scene understanding for applications such as 3D scene reconstruction, autonomous driving, and AR. 

### Encoder-Decoder Model Arch. for Creating Depth Estimation 
Existing solutions for depth estimation often produce blurry approximations of low resolution. This Repo presents a convolutional neural network for computing a high-resolution depth map given a single RGB image with the help of transfer learning. 

Following a standard encoder-decoder architecture, we leverage features extracted using high performing pre-trained networks when initializing our encoder along with augmentation and training strategies that lead to more accurate results.Even for a very simple decoder, our method is able to achieve detailed high-resolution depth maps.


### Global-Local Path Networks (GLPN) - Hugging Face

Global-Local Path Networks (GLPN) model trained on NYUv2 for monocular depth estimation.The novel arch. was introduced in the paper Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth by Kim et al. They deploy a hierarchical transformer encoder to capture and convey the global context, and design a lightweight yet powerful decoder to generate an estimated depth map while considering local connectivity.

By constructing connected paths between multi-scale local features and the global decoding stream with our proposed selective feature fusion module, the network can integrate both representations and recover fine details. In addition, the proposed decoder shows better performance than the previously proposed decoders, with considerably less computational complexity


### Dataset
We used a custom dataset child health monitoring system to estimate height of the child using depth images.There are two approach we have taken into account. 

- Using Transfer Learning approach 
- Using GLPN 


### Metrics
The most popular benchmark datasets are the KITTI and NYUv2 datasets. Models are typically evaluated using RMSE or absolute relative error.