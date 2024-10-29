<h2 align="center"> 💻 Detec_Obj_DeepLearning </h2>

# Image Classification Model

This repository contains code for an image classification model using a Convolutional Neural Network (CNN) with Keras and TensorFlow. The model is designed to identify and classify images into different classes.

<h4 align="center">Detec_Obj_DeepLearning 🚀</h4>

<div align="center">
    <img src="Img_Detec_Obj_DeepLearning/0_Inf_Proj.jpg" style="width: 45%; margin-right: 5%;" alt="0_Inf_Proj">
    <img src="Img_Detec_Obj_DeepLearning/1_result_loss_acurracy.png" style="width: 45%; margin-right: 5%;" alt="1_result_loss_acurracy">
  <br/>
  <br/>
   <img src="Img_Detec_Obj_DeepLearning/3_Detec_Obj_DeepLearning – treinamentoCNN_process.png" style="width: 45%; margin-right: 5%;" alt="3_Detec_Obj_DeepLearning – treinamentoCNN_process">
    <img src="Img_Detec_Obj_DeepLearning/1_result_loss_acurracy.png" style="width: 45%; margin-right: 5%;" alt="1_result_loss_acurracy">
   <br/>
  <br/>
   <img src="Img_Detec_Obj_DeepLearning/4_Detec_Obj_DeepLearning – treinamentoCNN_process.png" style="width: 45%; margin-right: 5%;" alt="4_Detec_Obj_DeepLearning – treinamentoCNN_process">
    <img src="Img_Detec_Obj_DeepLearning/5_Detec_Obj_DeepLearning – treinamentoCNN_process.png" style="width: 45%; margin-right: 5%;" alt="5_Detec_Obj_DeepLearning – treinamentoCNN_process">
</div>

---


## Prerequisites

To run this project, the following versions are required:

- TensorFlow version 2.0.0
- Keras version 2.3.1

### Key Libraries

- `os`
- `matplotlib`
- `cv2` (OpenCV)
- `numpy`
- `keras` and `tensorflow` for neural network modeling and training
- `sklearn` for training and test data splitting

## Main Parameters

- `path`: Directory where the sample images are stored.
- `batch_size_val`: Number of images processed in each training batch.
- `steps_per_epoch_val`: Number of iterations per epoch.
- `epochs_val`: Number of epochs for training.
- `imageDimensions`: Dimensions of the input images (32x32, RGB).

## Image Preprocessing

Images go through the following preprocessing steps:

1. **Grayscale conversion**.
2. **Histogram equalization** for intensity standardization.
3. **Normalization** to values between 0 and 1.

## Data Augmentation

An `ImageDataGenerator` is used to create variations of training images, enhancing data diversity and improving model generalization. Augmentation includes:

- Width and height shifts.
- Zoom.
- Angle adjustments.
- Rotation.

## Model Architecture

The CNN is structured as follows:

1. **Convolutional Layers** for feature extraction.
2. **MaxPooling Layers** for dimensionality reduction.
3. **Dropout Layer** for regularization.
4. **Dense Layer** for final classification.

**Activation Function:** `ReLU` for convolutional layers and `Softmax` for the output layer.  
**Optimizer:** `Adam` with a learning rate of 0.001.

## Training

The model is trained using `fit_generator`, which utilizes the training and validation datasets. Loss and accuracy history are recorded and displayed in plots at the end of training.

## Model Evaluation and Saving

The model is evaluated on the test set, with loss and accuracy scores printed at the end. The final model is saved to the `modelo.h5` file.

## Running the Code

1. Ensure the `Images` directory contains folders with images for each class.
2. Run the script to train the model.
3. View the performance charts at the end of training.

**Note**: Training time may vary between 20 to 30 minutes, depending on the machine.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

