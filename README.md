# DeepFake Detection System

A robust, explainable deep learning system for detecting manipulated (deepfake) images and videos using a combination of CNN-based feature extractors and Vision Transformer (ViT) modules.

## Features

- **Multiple Model Architectures**: CNN baseline (XceptionNet/EfficientNet), Vision Transformer (ViT), Temporal models (LSTM/GRU/Transformer), and Ensemble models
- **Frame-level and Video-level Detection**: Support for both frame-by-frame and temporal sequence analysis
- **Robustness Testing**: Evaluation against compression, resolution changes, blur, and adversarial attacks
- **Explainability**: Grad-CAM visualization for CNNs and attention maps for Transformers
- **Comprehensive Evaluation**: Multiple metrics including ROC-AUC, PR-AUC, EER, and TPR@FPR

## Project Structure

```
DEEPFAKE/
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── src/
│   ├── models/             # Model architectures
│   │   ├── cnn_model.py
│   │   ├── vit_model.py
│   │   ├── temporal_model.py
│   │   └── ensemble_model.py
│   ├── data/               # Dataset classes
│   │   └── dataset.py
│   ├── utils/              # Utilities
│   │   ├── config.py
│   │   ├── face_detection.py
│   │   └── preprocessing.py
│   ├── training/           # Training utilities
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/         # Evaluation utilities
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── explainability/     # Explainability tools
│       ├── gradcam.py
│       └── attention_viz.py
├── data/                   # Data directory (create this)
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── checkpoints/            # Model checkpoints
├── results/                # Evaluation results
└── logs/                   # Training logs
```

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create data directory structure:**
```bash
python setup_data_dirs.py
```

3. **Add your data:**
   - Place real videos/images in `data/train/real/`, `data/val/real/`, `data/test/real/`
   - Place fake videos/images in `data/train/fake/`, `data/val/fake/`, `data/test/fake/`

4. **Train a model:**
```bash
python train.py --config config.yaml
```

5. **Evaluate:**
```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DEEPFAKE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download face detection models (optional, for dlib):
   - Download `shape_predictor_68_face_landmarks.dat` from dlib and place in `models/` directory

4. Create data directories:
```bash
python setup_data_dirs.py
```

## Data Preparation

### Recommended Datasets

- **FaceForensics++**: Multiple manipulation methods and compression levels
- **DFDC (DeepFake Detection Challenge)**: Large, diverse videos
- **Celeb-DF**: More realistic deepfakes
- **FaceSwap / Face2Face / NeuralTextures**: Method variety

### Data Organization

Organize your data in the following structure:
```
data/
├── train/
│   ├── real/    # Real videos/images
│   └── fake/    # Fake videos/images
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`

## Configuration

Edit `config.yaml` to customize:
- Model architecture (CNN, ViT, Temporal, Ensemble)
- Training hyperparameters (learning rate, batch size, etc.)
- Data augmentation settings
- Evaluation metrics
- Robustness test parameters

## Usage

### Training

Train a model:
```bash
python train.py --config config.yaml
```

Resume training from checkpoint:
```bash
python train.py --config config.yaml --resume checkpoints/best_model.pth
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

Evaluate on specific test directory:
```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth --test_dir data/custom_test
```

## Model Architectures

### 1. CNN Baseline
- Uses pretrained EfficientNet or XceptionNet
- Good for texture-level artifact detection
- Fast inference

### 2. Vision Transformer (ViT)
- Captures global context and subtle inconsistencies
- Strong performance on high-quality deepfakes
- Attention visualization available

### 3. Temporal Model
- Processes video sequences
- Uses LSTM, GRU, or Transformer for temporal modeling
- Captures temporal inconsistencies (flicker, unnatural patterns)

### 4. Ensemble Model
- Combines CNN and ViT features
- Fusion methods: concatenation, addition, or gating
- Best overall performance

## Evaluation Metrics

The system computes:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **EER**: Equal Error Rate
- **TPR@FPR**: True Positive Rate at low False Positive Rates

## Robustness Testing

The system evaluates robustness to:
- **Compression**: JPEG compression at various quality levels
- **Resolution**: Different input resolutions
- **Blur**: Gaussian blur
- **Adversarial Attacks**: FGSM, PGD (if enabled)

## Explainability

### Grad-CAM for CNNs
Visualize which regions the CNN focuses on:
```python
from src.explainability import GradCAMExplainer

explainer = GradCAMExplainer(model)
cam = explainer.visualize(input_tensor, original_image, save_path='gradcam.png')
```

### Attention Visualization for ViT
Visualize attention maps from Transformer:
```python
from src.explainability import AttentionVisualizer

visualizer = AttentionVisualizer(model)
attention = visualizer.visualize_attention(input_tensor, original_image, save_path='attention.png')
```

## Training Tips

1. **Start with CNN baseline**: Fast to train, good baseline
2. **Use mixed precision**: Enable in config for faster training
3. **Monitor validation metrics**: Early stopping prevents overfitting
4. **Data augmentation**: Balance between realism and artifact preservation
5. **Cross-dataset evaluation**: Test on unseen datasets for generalization

## Ablation Studies

To perform ablation studies, modify `config.yaml` and train different configurations:
- CNN-only vs ViT-only vs Ensemble
- Frame-level vs Video-level
- With/without temporal modeling
- Different backbones (EfficientNet-B0 to B7)
- Impact of augmentation

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.yaml`
- Use smaller model (e.g., EfficientNet-B0 instead of B4)
- Enable gradient checkpointing

### Face Detection Fails
- Check face detection method in config
- Ensure faces are visible and not too small
- Try different face detectors (MTCNN, RetinaFace, OpenCV)

### Poor Performance
- Check data quality and balance
- Try different model architectures
- Adjust learning rate and training schedule
- Increase data augmentation

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{deepfake-detection-system,
  title={Robust DeepFake Detection using CNN and Vision Transformer},
  author={Your Name},
  year={2025}
}
```

## License

[Specify your license]

## Acknowledgments

- FaceForensics++ dataset
- Hugging Face Transformers
- PyTorch and timm libraries
- OpenCV and albumentations

## Future Work

- [ ] Audio branch for multimodal detection
- [ ] Real-time inference pipeline
- [ ] Web interface for easy testing
- [ ] Support for more datasets
- [ ] Advanced adversarial training
- [ ] Model compression and quantization

