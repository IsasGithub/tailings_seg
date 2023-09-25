# built under R version 4.2.2

### Segmentation of mining tailings in brazil
### some tailings are labelled as multiple polygons, i.e. predicted output should not always exactly match the goundtruth mask

# libraries
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

tensorflow::install_tensorflow()

# set wd
setwd("D:/DATEN ZWEI/Wue/SS_23/DeepLearning/Assignment/")

## IMAGE CONVERSION
# only necessary when using .tif files
# convert img tif from GEE to jpeg
# in.files <- list.files("./img_tif/","tif$")
# for(i in 0:(length(in.files)-1)){
#   tif <- terra::rast(paste0("./img_tif/", "img_", as.character(i), ".tif"))
#   x <- as.array(tif)
#   x <- array(x,c(448,448,3))
#   x <- apply(x,c(1,2,3),function(x){x*3})
#   x[,,c(3,1)] <- x[,,c(1,3)]
#   writeJPEG(x, target = paste0("./img/img_", as.character(i), ".jpeg"), quality = .7)
# }

# convert mask tif from GEE to jpeg
# in.files <- list.files("./mask_tif/","tif$")
# for(i in 0:(length(in.files)-1)){
#   tif <- raster(paste0("./mask_tif/", "mask_", as.character(i), ".tif"))
#   x <- raster2matrix(tif)
#   rotate <- function(x) t(apply(x, 2, rev))
#   x <- rotate(rotate(rotate((x))))
#   writeJPEG(x, target = paste0("./mask/mask_", as.character(i), ".jpeg"), quality = 1)
# }


## DATA PREPROCESSING
# loading data
data <- data.frame(
  img = list.files("./img/", full.names = T),
  mask = list.files("./mask/", full.names = T)
)

# split
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


# settle on a universal data type
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

###### augmentation
# augmentation 1
spectral_aug <- function(img){
  img <- tf$image$random_brightness(img, max_delta = .3)
  img <- tf$image$random_contrast(img, lower = .8, upper = 1.1)
  img <- tf$image$random_saturation(img, lower = .8, upper = 1.1)

  img <- tf$clip_by_value(img, 0, 1) 
}

# apply augmentation to all images
aug <- dataset_map(train_ds, function(x)
  list_modify(x, img = spectral_aug(x$img))
)

aug <- dataset_map(aug, function(x)
  list_modify(x,
              img = tf$image$flip_left_right(x$img),
              mask = tf$image$flip_left_right(x$mask)
  )
)

# double our original train dataset
train_ds_aug <- dataset_concatenate(train_ds, aug)

# augmentation 2: do the same as above, but flip up and down
aug <- dataset_map(train_ds, function(x)
  list_modify(x, img = spectral_aug(x$img))
)

aug <- dataset_map(aug, function(x)
  list_modify(x,
              img = tf$image$flip_up_down(x$img),
              mask = tf$image$flip_up_down(x$mask)
  )
)

# triple our original train dataset
train_ds_aug <- dataset_concatenate(train_ds_aug, aug)


# optional
# augmentation 3: flip left right AND up down, including random change of saturation, brightness and contrast
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
# quatruple our original dataset (original + augmentated data 1 + augmentated data 2)
train_ds_aug <- dataset_concatenate(train_ds_aug, aug)

######


# shuffeling
train_ds <- dataset_shuffle(train_ds, buffer_size = 1280)

# create batches
train_ds <- dataset_batch(train_ds, 10)

# unname dataset, remove all names and addresses
train_ds <- dataset_map(train_ds, unname)

## validation data set tensor slices
# create tensor slices
val_ds <- tensor_slices_dataset(testing(data))

# for looking at it in an R data model :
slices2list <- function(x) iterate(as_iterator(x))
val_ds_list <-  slices2list(val_ds)

## read and decode jpeg
val_ds <- dataset_map(val_ds, function(x){
  list_modify(x,
              img = tf$image$decode_jpeg(tf$io$read_file(x$img)),
              mask = tf$image$decode_jpeg(tf$io$read_file(x$mask)))
})

# settle on a universal data type
val_ds <- dataset_map(val_ds, function(x){
  list_modify(x,
              img = tf$image$convert_image_dtype(x$img,dtype = tf$float32),
              mask = tf$image$convert_image_dtype(x$mask,dtype = tf$float32))
})

# resize
val_ds <- dataset_map(val_ds, function(x){
  list_modify(
    x, 
    img = tf$image$resize(x$img, size = shape(input_shape[1], 
                                              input_shape[2])),
    mask = tf$image$resize(x$mask, size = shape(input_shape[1], 
                                                input_shape[2])))
})

# create batches
val_ds <- dataset_batch(val_ds, 10)

# unname dataset, remove all names and addresses
val_ds <- dataset_map(val_ds, unname)


## NETWORK DESIGN
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


## COMPILE THE MODEL
compile(
  unet_model,
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)

# train it
run <- fit(
  unet_model,
  train_ds,
  epochs = 8,
  validation_data = val_ds
)

## EVALUATION
# accuarcy and loss
evaluate(unet_model, val_ds)

## VISUALISATION
# extract prediction maps
pred_ds <- predict(unet_model, val_ds)
i = 13
img_path <- as.character(testing(data)[[i,1]])
mask_path <- as.character(testing(data)[[i,2]])
img <- magick::image_read(img_path)
mask <- magick::image_read(mask_path)
pred <- magick::image_read(as.raster(predict(object = unet_model,val_ds)[i,,,]))


out <- magick::image_append(c(
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE),
  magick::image_append(pred, stack = TRUE)
))

plot(out)

