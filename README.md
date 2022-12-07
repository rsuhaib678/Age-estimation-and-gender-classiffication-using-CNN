## Age estimation and gender classification using CNN
 
### Introduction

Given a subset of UTKFace dataset containing 5000 images. The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. The labels of each image are embedded in the file name, formatted as [age] [gender] [race] [date&time].jpg. We were supposed to use only first two labels, i.e, age(integer from 0 to 116), gender(0 for male and 1 for female). I have to perform some data pre-processing and data augmentation on the dataset like rescaling the pixels.
My goal was to build and train deep learning models to classify gender and age from the given dataset. The first model was supposed to be own model while the second model was an existing pre-trained CNN architecture on ImageNet and fine tune it as per the dataset.
At the end, I have plot the graph to compare the performance of both the models.

Dataset: https://susanqq.github.io/UTKFace/

![download (2)](https://user-images.githubusercontent.com/81761180/194522765-f31ca9d4-50c6-4e5f-a16a-5083fe0a5faf.png)


## Custom CNN
### Architecture

I have built a multi-output CNN taking a single input of shape (128 X 128 X 3). The input layer is followed by two convolutional layers based on a Conv2D layer having neurons each, with a ReLU activation, followed by a MaxPooling2D layer and then a BatchNormalization layer. It is followed by a final Conv2D layer and MaxPooling2D layer before disintegrating into two separate branches, one for age and other for gender.
The age branch starts with a Conv2D layer followed by a max_pooling2D layer then a BatchNormalization layer. The three layers are repeated again in the similar order. The final Conv2D layer is followed by a flatten layer and multiple dense and dropout layer. The gender branch follows the similar structure as that of age branch.

### Training

The model was trained for a batch size of 64 and number of epochs as 50. The training steps per epoch are 62 and the validation steps per epoch are 15. The optimizer used is Adam, loss function used for age is mean squared error and for gender is binary cross entropy, loss weights for age and gender are 0.1 and 1 respectively, metrics used for checking for performance of model are mae and accuracy. We tried different hyperparameters for our model to optimize its performance like changing the loss weights values and optimizer. Initially, I used learning rate of 0.004 but discarded it later due to no changes in performance.

### Performance

The performance of CNN was somewhat near to what we expected. Our goal was to reduce the age output mae below 7 and increase the goal output accuracy above 80%. The final loss value for weight function of [“Age”:0.1, “Gender”: 1] was 5.6032, age output mae = 4.3738, gender output accuracy of 95.56%, validation loss = 10.8850, validation age mae = 6.9310, validation gender accuracy = 86%.

### ResNet-50

For the pre-trained CNN model, I used ResNet-50, a 50 layers deep convolutional neural network. The pre-trained version of the model is trained on more than a million images from ImageNet database. It has 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer. It has 3.8 X 10^9 floating points operations.
Apart from these layers, I have introduced two separate branches to estimate gender and age. The gender and age branch consist of multiple dense and flatten layer. The final gender and age layer has an output shape of 1 unit.

### Training

The model was trained for a batch size of 64 and number of epochs as 50. The training steps per epoch are 62 and the validation steps per epoch are 15. The optimizer used is Adam, loss function used for age is mean squared error and for gender is binary cross entropy, loss weights for age and gender are 0.1 and 1 respectively, metrics used for checking for performance of model are mae and accuracy. We tried different hyperparameters for our model to optimize its performance like changing the loss weights values and optimizer. Initially, I used learning rate of 0.004 but discarded it later due to no changes in performance.

### Performance

The performance of CNN was somewhat near to what we expected. The goal was to reduce the age output mae below 7 and increase the goal output accuracy above 80%. Our final loss value for weight function of [“Age”:0.1, “Gender”: 1] was 5.6032, age output mae = 4.3738, gender output accuracy of 95.56%, validation loss = 10.8850, validation age mae = 6.9310, validation gender accuracy = 86%.
