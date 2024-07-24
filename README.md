# Skin-disease-classification

This project aims to classify various skin diseases using deep learning techniques. We've implemented two main approaches:

A custom Convolutional Neural Network (CNN) model
A transfer learning approach using ResNet50

Both models were trained to classify 19 different skin conditions.
Dataset

The dataset consists of images of 19 different skin diseases.
Images are preprocessed and normalized to a consistent size of 224x224 pixels.
The dataset is split into training, validation, and test sets.

Preprocessing
We implemented a custom preprocessing pipeline:

Resize images to 224x224 pixels.
Convert images to RGB if they're in a different color mode.
Normalize pixel values to the range [0, 1].

Model Architectures
1. Custom CNN Model
Our initial approach used a custom CNN architecture:

Multiple convolutional layers with increasing filter sizes
Max pooling layers for dimensionality reduction
Flatten layer to convert 2D feature maps to 1D vector
Dense layers for classification
Output layer with 19 units and softmax activation

2. Transfer Learning with ResNet50
To improve performance, we implemented a transfer learning approach:

Base model: ResNet50 (pre-trained on ImageNet)
Additional layers:

Global Average Pooling
Dense layer (1024 units, ReLU activation)
Output layer (19 units, Softmax activation)

Results

Custom CNN Model:

Best validation accuracy: 25%


ResNet50 Transfer Learning:

Best validation accuracy: 30%


Test set accuracy: 0.5557 (55.57%)


Challenges and Future Work

The current test accuracy with ResNet50 is relatively low (55.57%), indicating room for improvement.
Future steps include:

Analyzing the confusion matrix to identify problematic classes.
Checking for and addressing class imbalance.
Implementing careful data augmentation to increase dataset size without compromising image quality.
Fine-tuning the model architecture (e.g., unfreezing more layers, adding dropout).
Experimenting with different pre-trained models (e.g., EfficientNet, InceptionV3).
Investigating the impact of class imbalance and implementing techniques to address it.
