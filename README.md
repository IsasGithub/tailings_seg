# Segmentation of Mining Tailings in Brazil using DL: Testing Tensorflow in R and Python

## Introduction
This project explores the image segmentation of mining tailings from Sentinel-2 RGB images in Brazil using a DL approach. 

## Training Data
Our training data consists of 114 sets of each a Sentinel-2 imagery of mining tailings and its corresponding manually labelled segmentation mask. Both can be found in the folders `data/img` and `data/mask`. The training data was labelled in Google Earth with images of the year 2021, the training data was created using the provided GEE script `GEE_trainingsamples.txt`.

## Network Architecture
We implement a CNN with a simple U-net architechture using the Tenserflow API in R. Additionally we recreate the same network in Python to compare usability and reliaility. In total, the network has 10 convolutional layers, we half the image channels in each depth. We choose binary crossentropy as our loss function. Different learning rates and batch size parameters are tested.
![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/ddcaa1cb-099e-4337-b815-670bb21d8cf7)

## Results
### Tensorflow in R
This is an exemplary result of the implemented CNN in R, showing the labelled groundtruth mask on the left, the corresponding Sentinel-2 in the middle and the predicted image segmentation on the right:

![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/61203401-1569-4567-b1f8-b65bc255aad7)

We manage a validation accuracy of approx. 0.9, and validation loss of approx. 0.05.
![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/c289673f-b500-45d5-b983-58981775774c)

### Tensorflow in Python
This is an exemplary result of the implemented CNN in Python, showing the Sentinel-2 on the lefr, the corresponding groundtruth mask on the left in the middle and the predicted image segmentation on the right:
![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/10aea9be-8358-4474-9374-80a2d06cb417)
We have similar accuracy measures as with the network implemented in R.
![Figure_1](https://github.com/IsasGithub/tailings_seg/assets/116874799/00fbadb2-acac-43af-8b3a-ba31f2a05bfe)
![Figure_2](https://github.com/IsasGithub/tailings_seg/assets/116874799/68ca701f-c512-4eb2-9a89-cee85e4a576a)

| ![Figure_1](https://github.com/IsasGithub/tailings_seg/assets/116874799/00fbadb2-acac-43af-8b3a-ba31f2a05bfe) | ![Figure_2](https://github.com/IsasGithub/tailings_seg/assets/116874799/68ca701f-c512-4eb2-9a89-cee85e4a576a) |
| -- | --- |
| Training & Validation **Loss** | Training & Validation **Accuracy** |

***Figure 3:** Linear relationship between NDVI and AGB values for Sentinel and drone images.* 
