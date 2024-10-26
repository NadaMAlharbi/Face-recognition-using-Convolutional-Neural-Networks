# Face Recognition using Convolutional Neural Networks (CNN)

**Author:** Nada Mohammed Alharbi  
**Project Type:** Academic Project for CCAI-321: Artificial Neural Networks  
**Development Environment:** Kaggle Notebooks

## Project Overview
This project focuses on developing a facial recognition system using Convolutional Neural Networks (CNN). The model is trained on the ORL face dataset, also known as the Olivetti Face Dataset, which includes images of 40 individuals. Each individual has 10 images, resulting in a total of 400 images for training and testing.

The project involves multiple stages, including data loading, preprocessing, model training, and evaluation of its performance.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Tools and Libraries Used](#tools-and-libraries-used)
4. [Project Structure](#project-structure)
5. [Steps to Run the Project](#steps-to-run-the-project)
6. [Key Features](#key-features)
   - [Model Architecture](#model-architecture)
   - [Performance Evaluation](#performance-evaluation)
7. [Project Questions](#project-questions)
8. [Results and Observations](#results-and-observations)

## Dataset
- **Dataset Name:** ORL (Olivetti Face Dataset)
- **Total Images:** 400 images (40 individuals, 10 images per individual)
- **Data Shape:** Each image is 64x64 pixels, grayscale.

This dataset provides a structured and manageable number of images, making it suitable for experimenting with CNN models for face recognition tasks.

## Tools and Libraries Used
- **Languages:** Python
- **Libraries:**
  - `NumPy`
  - `Sklearn`
  - `Keras`
  - `Matplotlib`
  - `TensorFlow`
- **Platform:** Kaggle Notebooks (with GPU support for faster model training)

## Project Structure
The project contains the following files:
1. **`Face recognition using Convolutional Neural Networks.ipynb`:** The main Kaggle Notebook that contains the code for loading the dataset, training the CNN, and evaluating the results.

## Steps to Run the Project
To run the project on **Kaggle**, follow these steps:

1. **Environment Setup:**
   - Log in to your Kaggle account.
   - Ensure you have access to GPU resources for faster training (Kaggle automatically provides this for supported notebooks).
   - All necessary libraries (`NumPy`, `TensorFlow`, `Keras`, `Sklearn`, `Matplotlib`) are pre-installed in Kaggle Notebooks.

2. **Running the Code:**
   - Open the notebook `Face recognition using Convolutional Neural Networks.ipynb` on Kaggle.
   - Ensure that the necessary dataset files (`olivetti_faces.npy`, `olivetti_faces_target.npy`) are uploaded into your Kaggle environment if they are not already included.
   - Run the notebook cells to load the dataset, preprocess it, and train the CNN model.

3. **Evaluation:**
   - After training, the notebook will display the model's performance in terms of training and validation accuracy.
   - The notebook includes visualizations of results (e.g., accuracy and loss graphs) using Matplotlib.

## Key Features
### Model Architecture
- The CNN model includes multiple convolutional layers, pooling layers, and fully connected layers.
- The model uses **ReLU** activation and **softmax** for classification across the 40 classes (one for each individual).

### Performance Evaluation
- The training process includes accuracy and validation loss measurements over 10 epochs.
- Modifications in the pooling size (from (4,4) to (2,2)) are explored to improve the model's performance.

## Project Questions
The project also explores key questions about CNN model performance, such as:
1. **What is the output of `model.summary()`?**
2. **What is the initial training and validation accuracy?**
3. **How do pooling layers affect validation accuracy?**
4. **How do changes in the batch size and number of epochs affect the model's performance?**

## Results and Observations
- The model's performance can be improved by adjusting hyperparameters such as pooling layer size, batch size, and number of epochs.
- The ORL face dataset is relatively small, but it provides a good starting point for facial recognition tasks using CNNs.
