# Olivetti Faces — Unsupervised and Semi-Supervised Learning

## Overview

This repository contains a structured mini-project based on **Exercises 10–13 of Chapter 8 (Unsupervised Learning Techniques)** from *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

The project focuses on applying **clustering, dimensionality reduction, Gaussian mixture models, and anomaly detection** to the classic **Olivetti Faces dataset**, with the goal of developing a deep, practical understanding of unsupervised and semi-supervised learning techniques.

This repository is designed to demonstrate not only correct usage of machine learning algorithms, but also **sound experimental design, evaluation practices, and clear reasoning**, in a way suitable for an internship-level GitHub portfolio.

---

## Dataset

The Olivetti Faces dataset consists of:

- 400 grayscale face images
- Image resolution: 64 × 64 pixels
- Each image flattened into a 4,096-dimensional vector
- 40 different individuals, with 10 images per person
- Pixel values already scaled to the range [0, 1]

The dataset is loaded using: sklearn.datasets.fetch_olivetti_faces

Stratified splitting is used to ensure that each person is equally represented in the training, validation, and test sets.

---

## Project Goals

The main goals of this project are to:

1. Explore clustering structure in high-dimensional image data
2. Use k-means clustering for data exploration and representation learning
3. Apply clustering as a dimensionality reduction technique for classification
4. Train Gaussian Mixture Models for generation and density estimation
5. Detect anomalies using probabilistic models and reconstruction error
6. Connect theory from Chapter 8 to concrete, reproducible experiments

---

## Methods and Techniques

The following techniques are used throughout the notebook:

- **k-means clustering**
  - Elbow method and silhouette score for selecting the number of clusters
  - Visualization of cluster contents (face images)
  - k-means used as a feature transformation method

- **Supervised classification**
  - Baseline classifier trained on original features
  - Classifier trained on k-means-transformed features
  - Feature augmentation using original + reduced representations

- **Dimensionality reduction**
  - Principal Component Analysis (PCA)
  - PCA preserving 99% of variance for efficiency and stability

- **Gaussian Mixture Models (GMM)**
  - Training in PCA-reduced space
  - Model selection considerations
  - Generating new synthetic face images
  - Density-based anomaly detection using likelihood scores

- **Anomaly detection**
  - Detection using GMM density thresholds
  - Detection using PCA reconstruction error
  - Comparison between normal and artificially modified images

---

## Structure of the Repository 

olivetti-unsupervised-learning/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_olivetti_unsupervised_learning.ipynb
├── src/
│   ├── utils.py
│   ├── visualization.py
│   └── metrics.py
├── figures/
│   ├── kmeans_clusters.png
│   ├── silhouette_elbow.png
│   ├── gmm_generated_faces.png
│   └── anomaly_examples.png
└── .gitignore

---

## Key Results

- k-means clustering groups faces by visual similarity (pose, lighting, expression), not by identity
- Using k-means as a feature transformation can improve classifier performance when k is chosen carefully
- PCA significantly reduces dimensionality while preserving meaningful facial structure
- Gaussian Mixture Models can generate realistic face-like images in reduced space
- Both GMM density scores and PCA reconstruction error are effective for anomaly detection
- Reconstruction-based methods are particularly intuitive for image data

---

## Reproducibility

- All experiments use fixed random seeds
- The dataset is publicly available through scikit-learn
- Dependencies are listed in `requirements.txt`
- The notebook runs end-to-end without manual intervention

---

## Learning Outcome

This project demonstrates that unsupervised learning methods are not only exploratory tools, but can also:

- Improve supervised learning through better representations
- Enable data-efficient learning
- Support anomaly and novelty detection
- Provide insight into complex, high-dimensional datasets

The experiments confirm the main message of Chapter 8:  
**unsupervised learning is powerful, but its effectiveness depends heavily on assumptions, data structure, and careful evaluation.**

---

## References

- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- Scikit-learn documentation: https://scikit-learn.org

## Author

Created by Nur
Machine Learning Student and aspiring ML Engineer
