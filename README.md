# Segmentation of Mining Tailings in Brazil using DL: Testing TensorFlow in R and Python

## Introduction
The demand for mineral resources is increasing sharply as substances such as copper, nickel, cobalt and graphite are necessary components in many clean energy technologies. This rising demand can be seen in the study case of Brazil, where the mining sectors' revenue is constantly growing. Tailings are a by-product of those mining activities, consisting of left-overs from rock or soil in which mined minerals occur. Monitoring of these mining tailings is essential for responsible management as they can pose to be potential risks to human health and safety, the environment and surrounding infrastructure. In this project, we explore the possibility of not only detecting but segmenting mining tailings from Sentinel-2 RGB images in Brazil using a DL approach. We implement the network using the open-source software library TensorFlow with a plugin in R and in Python to compare usability and reliability.

## Training Data
Our training data consists of 114 sets of each a Sentinel-2 imagery of mining tailings and its corresponding manually labelled segmentation mask. Both can be found in the folders `data/img` and `data/mask`. The training data was labelled in Google Earth with images of the year 2021, the training data was created using the provided GEE script `GEE_trainingsamples.txt`.

## Network Architecture
We implement a CNN with a simple U-net architechture using the TensorFlow API in R. Additionally we recreate the same network in Python. In total, the network has 10 convolutional layers, we half the image channels in each depth. We choose binary crossentropy as our loss function. Different learning rates and batch size parameters are tested.

![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/ddcaa1cb-099e-4337-b815-670bb21d8cf7)

***Figure 1:** The implemented U-net architecture.* 

## Results
### TensorFlow in R
This is an exemplary result of the implemented CNN in R, showing the labelled groundtruth mask on the left, the corresponding Sentinel-2 in the middle and the predicted image segmentation on the right:

![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/61203401-1569-4567-b1f8-b65bc255aad7)

***Figure 2:** Exemplary result of predicted image segmentation (R).* 


We manage a validation accuracy of approx. 0.9, and validation loss of approx. 0.05.
| ![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/44087c93-9035-4088-89ca-1c5baac0bd45) | ![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/af773968-6810-435a-944f-dbf1d91ffa77)|
| -- | --- |
| Training & Validation **Loss** | Training & Validation **Accuracy** |

***Figure 3:** Loss and accuracies achieved in R.* 

### TensorFlow in Python
This is an exemplary result of the implemented CNN in Python, showing the Sentinel-2 on the left, the corresponding groundtruth mask on the left in the middle and the predicted image segmentation on the right:
![image](https://github.com/IsasGithub/tailings_seg/assets/116874799/10aea9be-8358-4474-9374-80a2d06cb417)

***Figure 4:** Exemplary result of predicted image segmentation (Python).* 


We have similar accuracy measures as with the network implemented in R.
| ![Figure_1](https://github.com/IsasGithub/tailings_seg/assets/116874799/00fbadb2-acac-43af-8b3a-ba31f2a05bfe) | ![Figure_2](https://github.com/IsasGithub/tailings_seg/assets/116874799/68ca701f-c512-4eb2-9a89-cee85e4a576a)|
| -- | --- |
| Training & Validation **Loss** | Training & Validation **Accuracy** |

***Figure 3:** Loss and accuracies achieved in Python.* 

## Small Comparison of TensorFlow in R and Python
We faced some difficulties in getting the TensorFlow plugin for R running. This was the initial reason for implementing the same network in Python as well. While we managed to get the R script running and aquired some pretty good results, the Python community provides more support and more functionalities of TensorFlow are available, e.g. decoding .tif-files. We tested a few different network architectures and parameter settings, however we were slighly limited to our available hardware and computational power. As future work, we could try running the network in a cluster setup to further improve parameter tuning. Additionally we could work on the amount and quality of the training data. Nonetheless, this project gives an idea on the feasability of image segmentation of mining tailings and we learnt a lot about dos and donts and having fun testing things in DL. 🐘
