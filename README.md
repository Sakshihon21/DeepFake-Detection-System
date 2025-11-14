
🛡️ DeepFake Detection System
A modern, high-accuracy DeepFake Detection System built using Vision Transformers (ViT), EfficientNetV2, and MediaPipe.
This project provides a complete deepfake forensics pipeline — from dataset preparation to training, evaluation, and visualization.

✨ Key Features
Advanced Models:
Uses ViT and EfficientNetV2 for robust spatial feature learning.

MediaPipe Face Extraction:
Avoids heavy dependencies like dlib and performs fast, reliable face detection.

Modular Pipeline:
Clean structure for preprocessing, training, evaluation, and visualization.

Explainability Tools:
Provides Grad-CAM / attention maps for understanding model decisions.

Modern Training Setup:
Mixed precision, One-Cycle LR, augmented datasets, and reproducible configs.

🚀 Project Structure
powershell
Copy code
DeepFake-Detection-System/
│── src/
│   ├── data/                     # Dataset utilities
│   ├── models/                   # ViT, EfficientNetV2, ensemble code
│   ├── utils/                    # Preprocessing, augmentation, helpers
│   └── inference/                # Inference scripts
│
│── train.py                      # Main training pipeline
│── evaluate.py                   # Evaluation + metrics
│── visualize_explanations.py     # Grad-CAM / attention visualizations
│── setup_data_dirs.py            # Dataset directory builder
│── test_setup.py                 # Quick environment test
│── config.yaml                   # Central configuration
│── requirements.txt              # Python dependencies
│── README.md                     # Project documentation
📦 Installation
bash
Copy code
git clone https://github.com/Sakshihon21/DeepFake-Detection-System.git
cd DeepFake-Detection-System
pip install -r requirements.txt
🧩 Dataset Preparation
Use the utility script to create dataset folders:

bash
Copy code
python setup_data_dirs.py
Add your deepfake dataset (FaceForensics++, DFDC, CelebDF, etc.) into the designated folders.

🏋️ Training the Model
Train ViT Model
bash
Copy code
python train.py --model vit
Train EfficientNetV2 Model
bash
Copy code
python train.py --model efficient
📈 Evaluation
bash
Copy code
python evaluate.py --model vit
You’ll get metrics such as:
✔ Accuracy
✔ Precision, Recall, F1
✔ Confusion Matrix
✔ ROC-AUC

🔍 Explainability (Grad-CAM / Attention)
bash
Copy code
python visualize_explanations.py --image path/to/test.jpg
🎯 Inference (Detect DeepFake on a Single Image)
bash
Copy code
python src/inference/predict.py --image test.jpg
🧠 Future Improvements
Add Temporal 3D CNN for video-level deepfake detection

Add hybrid ViT-CNN temporal fusion

Deploy FastAPI + Streamlit web interface

Add synthetic data generation using diffusion models

