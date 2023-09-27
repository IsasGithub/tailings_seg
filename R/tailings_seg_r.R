### Segmentation of Mining Tailings in Brazil
### Authors: Isabella Metz and Sunniva McKeever
### Built under R version 4.3.1


# Load libraries
library(terra)
library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(stars)
library(raster)
library(reticulate)
library(mapview)
library(magick)
library(oceanmap)
library(jpeg)
library(dplyr)
library(spatialEco)

# please set the path to your wd
setwd("D:/Isa/DeepLearning/Assignment/")


## convert img tif from GEE to jpeg
# only necessary tif-files still have to be converted

# in.files <- list.files("./img_tif/","tif$")
# for(i in 0:(length(in.files)-1)){
#   tif <- terra::rast(paste0("./img_tif/", "img_", as.character(i), ".tif"))
#   x <- as.array(tif)
#   x <- array(x,c(448,448,3))
#   x <- apply(x,c(1,2,3),function(x){x*3})
#   x[,,c(3,1)] <- x[,,c(1,3)]
#   writeJPEG(x, target = paste0("./img/img_", as.character(i), ".jpeg"), quality = .7)
# }
 
# # convert mask tif from GEE to jpeg
# in.files <- list.files("./mask_tif/","tif$")
# for(i in 0:(length(in.files)-1)){
#   tif <- raster(paste0("./mask_tif/", "mask_", as.character(i), ".tif"))
#   x <- raster2matrix(tif)
#   rotate <- function(x) t(apply(x, 2, rev))
#   x <- rotate(rotate(rotate((x))))
#   writeJPEG(x, target = paste0("./mask/mask_", as.character(i), ".jpeg"), quality = 1)
# }


# loading data
data <- data.frame(
  img = list.files("./img/", full.names = T),
  mask = list.files("./mask/", full.names = T)
)

# split data
data <- initial_split(data, prop = .75)

# input shape and batch size
input_shape <- c(448,448,3)
batch_size <- 3


## train data set tensor slices

# create tensor slices
train_ds <- tensor_slices_dataset(training(data))

# read and decode jpegs
train_ds <- dataset_map(train_ds, function(x){
  list_modify(x, 
              img = tf$image$decode_jpeg(tf$io$read_file(x$img)),
              mask = tf$image$decode_jpeg(tf$io$read_file(x$mask)))
})


# settle universal data type
train_ds <- dataset_map(train_ds, function(x){
  list_modify(x, 
              img = tf$image$convert_image_dtype(x$img,dtype = tf$float32),
              mask = tf$image$convert_image_dtype(x$mask,dtype = tf$float32))
})

# resize
train_ds <- dataset_map(train_ds, function(x){
  list_modify(
    x, 
    img = tf$image$resize(x$img, size = shape(input_shape[1], 
                                              input_shape[2])),
    mask = tf$image$resize(x$mask, size = shape(input_shape[1], 
                                                input_shape[2])))
})

## image augmentation
# first augmentation
spectral_aug <- function(img){
  img <- tf$image$random_brightness(img, max_delta = .3)
  img <- tf$image$random_contrast(img, lower = .8, upper = 1.1)
  img <- tf$image$random_saturation(img, lower = .8, upper = 1.1)

  img <- tf$clip_by_value(img, 0, 1) 
}

aug <- dataset_map(train_ds, function(x)
  list_modify(x, img = spectral_aug(x$img))
) # apply to all images

aug <- dataset_map(aug, function(x)
  list_modify(x,
              img = tf$image$flip_left_right(x$img),
              mask = tf$image$flip_left_right(x$mask)
  )
)

train_ds_aug <- dataset_concatenate(train_ds, aug) # doubles amount of training data


# augmentation 2: same as above but flip up and down
aug <- dataset_map(train_ds, function(x)
  list_modify(x, img = spectral_aug(x$img))
)

aug <- dataset_map(aug, function(x)
  list_modify(x,
              img = tf$image$flip_up_down(x$img),
              mask = tf$image$flip_up_down(x$mask)
  )
)

train_ds_aug <- dataset_concatenate(train_ds_aug, aug) # triples


# augmentation 3: flip left right and up down
aug <- dataset_map(train_ds, function(x)
  list_modify(x, img = spectral_aug(x$img))
)

aug <- dataset_map(aug, function(x)
  list_modify(x, img = tf$image$flip_left_right(x$img),
              mask = tf$image$flip_left_right(x$mask))
)

aug <- dataset_map(aug, function(x)
  list_modify(x, img = tf$image$flip_up_down(x$img),
              mask = tf$image$flip_up_down(x$mask))
)
train_ds_aug <- dataset_concatenate(train_ds_aug, aug) # quadruples


## further data preprocessing
# shuffeling
train_ds <- dataset_shuffle(train_ds, buffer_size = 1280)

# create batches
train_ds <- dataset_batch(train_ds, 10)

# unname dataset
train_ds <- dataset_map(train_ds, unname)

## validation data set tensor slices
# same as for training data
val_ds <- tensor_slices_dataset(testing(data))
slices2list <- function(x) iterate(as_iterator(x))
val_ds_list <-  slices2list(val_ds)

val_ds <- dataset_map(val_ds, function(x){
  list_modify(x,
              img = tf$image$decode_jpeg(tf$io$read_file(x$img)),
              mask = tf$image$decode_jpeg(tf$io$read_file(x$mask)))
})

val_ds <- dataset_map(val_ds, function(x){
  list_modify(x,
              img = tf$image$convert_image_dtype(x$img,dtype = tf$float32),
              mask = tf$image$convert_image_dtype(x$mask,dtype = tf$float32))
})

val_ds <- dataset_map(val_ds, function(x){
  list_modify(
    x, 
    img = tf$image$resize(x$img, size = shape(input_shape[1], 
                                              input_shape[2])),
    mask = tf$image$resize(x$mask, size = shape(input_shape[1], 
                                                input_shape[2])))
})

val_ds <- dataset_batch(val_ds, 10)

val_ds <- dataset_map(val_ds, unname)


## network design
l2 <- 0.01 # weight decay 
input_tensor <- layer_input(shape = input_shape)

# contracting path
# cov block 1
unet_tensor <- layer_conv_2d(
  input_tensor, filters = 64, 
  kernel_size = c(3,3), 
  padding = "same", 
  activation = "relu",
  kernel_regularizer = regularizer_l2(l2)
)

conc_tensor2 <- layer_conv_2d(
  unet_tensor, filters = 64,
  kernel_size = c(3,3), 
  padding = "same", 
  activation = "relu",
  kernel_regularizer = regularizer_l2(l2)
)

unet_tensor <- layer_max_pooling_2d(conc_tensor2)


# cov block 2
unet_tensor <- layer_conv_2d(
  unet_tensor, filters = 128, 
  kernel_size = c(3,3), 
  padding = "same", 
  activation = "relu",
  kernel_regularizer = regularizer_l2(l2)
)

conc_tensor1 <- layer_conv_2d(
  unet_tensor, filters = 128,
  kernel_size = c(3,3), 
  padding = "same", 
  activation = "relu",
  kernel_regularizer = regularizer_l2(l2)
)

unet_tensor <- layer_max_pooling_2d(conc_tensor1)


# bottom curve
unet_tensor <- layer_conv_2d(
  unet_tensor, filters = 256, 
  kernel_size = c(3,3), 
  padding = "same", 
  activation = "relu",
  kernel_regularizer = regularizer_l2(l2)
)

unet_tensor <- layer_conv_2d(
  unet_tensor, filters = 256, 
  kernel_size = c(3,3), 
  padding = "same", 
  activation = "relu",
  kernel_regularizer = regularizer_l2(l2)
)


# expanding path
# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(
  unet_tensor, filters = 128,
  kernel_size = c(2,2), strides = 2,
  padding = "same", kernel_regularizer = regularizer_l2(l2)
)

unet_tensor <- layer_concatenate(list(conc_tensor1, unet_tensor))

unet_tensor <- layer_conv_2d(
  unet_tensor, filters = 128,
  kernel_size = c(3,3),
  padding = "same", activation= "relu", kernel_regularizer = regularizer_l2(l2)
)

unet_tensor <- layer_conv_2d(
  unet_tensor, filters = 128,
  kernel_size = c(3,3),
  padding = "same", activation= "relu", kernel_regularizer = regularizer_l2(l2)
)

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(
  unet_tensor, filters = 64,
  kernel_size = c(2,2), strides = 2,
  padding = "same", kernel_regularizer = regularizer_l2(l2)
)

unet_tensor <- layer_concatenate(list(conc_tensor2, unet_tensor))

unet_tensor <- layer_conv_2d(
  unet_tensor, filters = 64,
  kernel_size = c(3,3),
  padding = "same", activation = "relu", kernel_regularizer = regularizer_l2(l2)
)

unet_tensor <- layer_conv_2d(
  unet_tensor, filters = 64,
  kernel_size = c(3,3),
  padding = "same", activation = "relu", kernel_regularizer = regularizer_l2(l2)
)

# output
unet_tensor <-  layer_conv_2d(
  unet_tensor, filter = 1, kernel_size = 1,
  activation = "sigmoid"
)

unet_model <- keras_model(inputs = input_tensor, output = unet_tensor)


## compile model
compile(
  unet_model,
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)

# train model
run <- fit(
  unet_model,
  train_ds,
  epochs = 8,
  validation_data = val_ds
)


## evaluation
evaluate(unet_model, val_ds)

## run model on validation data
pred_ds <- predict(unet_model, val_ds)

## invert values
# our model still inverts the predicted probabilities, future work to figure out why, until then...
dim1 <-  29
dim2 <- 448
dim3 <- 448

for (i in 1:dim1){
  for(j in 1:dim2){
    for(k in 1:dim3){
      pred_ds[i,j,k,1] <- (1 - pred_ds[i,j,k,1])
    }
  }
}


## visualisation
i = 19
img_path <- as.character(testing(data)[[i,1]])
mask_path <- as.character(testing(data)[[i,2]])
img <- magick::image_read(img_path)
mask <- magick::image_read(mask_path)
pred <- magick::image_read(as.raster(pred_ds[i,,,]))

out <- magick::image_append(c(
  
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE),
  magick::image_append(pred, stack = TRUE)
)
)

plot(out)