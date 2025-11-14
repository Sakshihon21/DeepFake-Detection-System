DeepFake Detection System

A deep learning project designed to detect manipulated or fake videos using Vision Transformers (ViT), EfficientNetV2, and MediaPipe-based face extraction.

📌 Overview

This project focuses on identifying DeepFake content by combining modern deep neural network architectures with efficient preprocessing methods.
It includes end-to-end modules for dataset preparation, preprocessing, training, evaluation, and inference.

✨ Key Features

Face Extraction using MediaPipe (Fast, no dlib required)

Two Strong Models: Vision Transformer & EfficientNetV2

Ensemble-Ready Architecture for better accuracy

Clean & Modular Codebase (easy to customize and extend)

Training & Evaluation Scripts included

Future-ready for video-level fake detection

📂 Project Structure
DeepFake-Detection-System/
├── data/                      # Dataset folders (real/fake)
├── models/                    # ViT, EfficientNetV2, Ensemble
├── utils/                     # Face extraction, augmentations
├── inference/                 # Prediction scripts
├── train.py                   # Train models
├── evaluate.py                # Evaluate trained models
├── requirements.txt           # Dependencies
└── README.md                  # Documentation

🚀 Getting Started
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Prepare Dataset

Organize your dataset as:

data/
 ├── real/
 ├── fake/


Use the provided preprocessing script for face extraction.

3️⃣ Train a Model
python train.py --model vit


python train.py --model efficientnet

4️⃣ Evaluate the Model
python evaluate.py --model vit

5️⃣ Run Inference
python inference/predict.py --image path/to/image.jpg

🎯 Future Enhancements

Video-level temporal modeling (3D CNN / LSTM)

Ensemble of ViT + EfficientNet + Temporal CNN

Streamlit/FastAPI demo interface

Diffusion-based synthetic training data
