# Segmentation of Mining Tailings in Brazil using DL in R

This project explores the image segmentation of mining tailings from Sentinel-2 RGB images in Brazil using a DL approach. We implement a CNN with a simple U-net architechture using the Tenserflow API in R. In total, the network has 10 convolutional layers.

Our training data consists of 114 sets of each a Sentinel-2 imagery of mining tailings and its corresponding manually labelled segmentation mask. Both can be found in the folders `data/img` and `data/mask`. The training data was labelled in Google Earth with images of the year 2021, the training data was created using the provided GEE script `GEE_trainingsamples.txt`.

This is an exemplary result of the implemented CNN, showing the labelled groundtruth mask on the left, the corresponding Sentinel-2 in the middle and the predicted image segmentation on the right:

![result_1](https://github.com/IsasGithub/tailings_seg/assets/116874799/2acbd300-4c5c-456e-8282-7cddd56f79f2)
