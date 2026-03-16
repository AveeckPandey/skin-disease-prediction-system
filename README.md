Skin Disease Prediction System

AI-powered web application for detecting and classifying skin diseases from images using deep learning and explainable AI.

The system allows users to upload a skin image and receive a prediction along with visual explanations showing which regions of the image influenced the model’s decision.

Project Overview

Skin diseases often require early identification for proper treatment. This project demonstrates how deep learning and computer vision can assist in identifying common dermatological conditions using image classification models.

The system integrates a trained convolutional neural network with a full-stack web application to deliver predictions in real time.

Features

Image-based skin disease classification

Real-time prediction via web interface

Confidence score for predictions

Explainable AI visualizations

Grad-CAM heatmaps for CNN interpretation

SHAP-based feature attribution

REST API backend for model inference

React-based interactive frontend

Diseases Detected

The system currently detects the following conditions:

BA-cellulitis

BA-impetigo

FU-athlete-foot

FU-nail-fungus

FU-ringworm

PA-cutaneous-larva-migrans

VI-chickenpox

VI-shingles

Tech Stack
Frontend

React.js

Axios

Backend

Flask API

Flask-CORS

Machine Learning

TensorFlow

Keras

MobileNet-based CNN

Explainable AI

Grad-CAM

SHAP

Image Processing

User Upload Image
        │
        ▼
React Frontend
        │
        ▼
Flask API Backend
        │
        ▼
Image Preprocessing
        │
        ▼
CNN Model (TensorFlow)
        │
        ▼
Prediction + Confidence
        │
        ▼
Explainability Layer
   ├── Grad-CAM
   └── SHAP
        │
        ▼
API Response
        │
        ▼
Frontend Visualization

skin-disease-prediction-system
│
├── backend
│   ├── app.py
│   ├── model
│   │   └── skin_disease_mobilenet_model.h5
│   └── requirements.txt
│
├── frontend
│   ├── src
│   └── package.json
│
├── dataset
│
├── notebooks
│   └── model_training.ipynb
│
└── README.md

OpenCV

NumPy
