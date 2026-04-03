# Problem Set [01] - [Pneumonia Detection in Pediatric Chest X-Rays using a Convolutional Neural Network (CNN)]

## 1. Approach
The goal of this project is to build an automated system capable of classifying Chest X-Ray images into two categories: Normal and Pneumonia.

My approach involved:
* Data Organization: Utilizing a structured directory format (train/, val/, and test/) to ensure the model generalizes well to unseen data.

* Image Preprocessing: Normalizing pixel values to a range of [0, 1] to improve convergence during training.

* Binary Classification: Since there are only two classes, I used a Sigmoid activation function in the final layer paired with Binary Crossentropy loss.

## 2. Methodology
The solution is implemented using TensorFlow/Keras and follows these technical steps:
* Data Augmentation: To prevent overfitting and expand the dataset's diversity, I used ImageDataGenerator to apply random shears, zooms, and horizontal flips to the training images.

* Model Architecture: I designed a Sequential Convolutional Neural Network (CNN) consisting of:

1.Three Convolutional Layers (Conv2D) with increasing filters (32, 64, 128) to extract hierarchical features.

2.MaxPooling layers to reduce spatial dimensions and computational load.

3.A Flatten layer followed by a Dense hidden layer (128 units) with ReLU activation.

4.A Dropout (0.5) layer to further reduce overfitting by randomly deactivating neurons during training.

* Optimization: The model was compiled using the Adam optimizer, which adaptively adjusts the learning rate.

* Evaluation: Beyond simple accuracy, I implemented a Classification Report (Precision, Recall, and F1-Score) using sklearn to understand the model's performance on both classes specifically.

## 3. Findings
* Model Training: The model was trained over 10 epochs, showing a steady decrease in loss and an increase in validation accuracy.

* Test Performance: The final accuracy on the test set is printed in the console (e.g., Test Accuracy: XX.XX%).

* Classification Balance: By analyzing the classification report, it was possible to determine if the model is more biased toward "Normal" or "Pneumonia" cases, which is critical in a medical diagnostic context where false negatives must be minimized.

* Deployment Ready: The final trained weights are saved as pneumonia_detection_model.h5 for future inference or deployment.


