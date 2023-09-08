# Depth-Estimation
Monocular Depth Estimation is the task of estimating the depth value (distance relative to the camera) of each pixel given a single (monocular) RGB image. This challenging task is a key prerequisite for determining scene understanding for applications such as 3D scene reconstruction, autonomous driving, and AR. 

### Encoder-Decoder Model Arch. for Creating Depth Estimation 
Existing solutions for depth estimation often produce blurry approximations of low resolution. This Repo presents a convolutional neural network for computing a high-resolution depth map given a single RGB image with the help of transfer learning. 

Following a standard encoder-decoder architecture, we leverage features extracted using high-performing pre-trained networks when initializing our encoder along with augmentation and training strategies that lead to more accurate results. Even for a very simple decoder, our method is able to achieve detailed high-resolution depth maps.


### Global-Local Path Networks (GLPN) - Hugging Face

Global-Local Path Networks (GLPN) model trained on NYUv2 for monocular depth estimation. The novel arch. was introduced in the paper Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth by Kim et al. They deploy a hierarchical transformer encoder to capture and convey the global context and design a lightweight yet powerful decoder to generate an estimated depth map while considering local connectivity.

By constructing connected paths between multi-scale local features and the global decoding stream with our proposed selective feature fusion module, the network can integrate both representations and recover fine details. In addition, the proposed decoder shows better performance than the previously proposed decoders, with considerably less computational complexity


### Dataset
We used a custom dataset child health monitoring system to estimate the height of the child using depth images. The dataset contains Child Images from multiple angles and it contains a Depth camera image and an RGB image along with the camera intrinsics details mainly fx , fy , cx , cy - this will be helpful in generating NERF image at later stage . 


### Model Training 
There are two approaches : 

- Using the Transfer Learning approach 
- Using GLPN

#### Transfer Learning Approach 

With the Transfer Learning Approach we used existing Pre-trained Models from Kitty dataset having Monocular Depth images and we use the pretrained network to learn the features of new dataset.The approach works great in creating a depth map but we were not able to improve the accuracy beyond a threshould . 


### Hugging Face GLPN

The State of the Art GLPN deploy a hierarchical transformer encoder to capture and convey the global context and design a lightweight yet powerful decoder to generate an estimated depth map while considering local connectivity. Thus creating a better depth map image than Transfer learning approach used in previous method 


### Metrics
The most popular benchmark datasets are the KITTI and NYUv2 datasets. Models are typically evaluated using RMSE or absolute relative error.
