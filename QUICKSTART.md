# Quick Start Guide

This guide will help you get started with the DeepFake Detection System quickly.

## Step 1: Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

**Note:** For face detection with dlib, you may need to install dlib separately:
```bash
# On Windows (if pip install fails)
# Download dlib wheel from: https://github.com/sachadee/Dlib
pip install dlib-19.24.0-cp39-cp39-win_amd64.whl  # Adjust version
```

## Step 2: Prepare Data

```bash
# Create data directory structure
python setup_data_dirs.py
```

Then organize your data:
- **Training data**: `data/train/real/` and `data/train/fake/`
- **Validation data**: `data/val/real/` and `data/val/fake/`
- **Test data**: `data/test/real/` and `data/test/fake/`

### Download Datasets

Recommended datasets:
1. **FaceForensics++**: https://github.com/ondyari/FaceForensics
2. **DFDC**: https://www.kaggle.com/c/deepfake-detection-challenge
3. **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics

## Step 3: Configure

Edit `config.yaml` to set:
- Model architecture (start with `cnn` for faster training)
- Batch size (reduce if GPU memory is limited)
- Learning rate
- Data paths

**Example for first run:**
```yaml
model:
  name: "cnn"  # Start with CNN, faster to train
  cnn_backbone: "efficientnet_b0"  # Smaller model

training:
  batch_size: 8  # Reduce if OOM
  num_epochs: 20  # Start with fewer epochs
```

## Step 4: Train

```bash
python train.py --config config.yaml
```

**Monitor training:**
- Checkpoints saved in `checkpoints/`
- Best model: `checkpoints/best_model.pth`
- Training history: `checkpoints/training_history.json`

**Tips:**
- Start with a small subset of data to test the pipeline
- Use smaller models (EfficientNet-B0) for faster iteration
- Monitor validation loss to avoid overfitting

## Step 5: Evaluate

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

Results will be saved in `results/`:
- `metrics.json`: All evaluation metrics
- `confusion_matrix.png`: Confusion matrix
- `roc_curve.png`: ROC curve
- `pr_curve.png`: Precision-Recall curve

## Step 6: Visualize Explanations

```bash
python visualize_explanations.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/test_image.jpg \
    --output explanation.png
```

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution:**
- Reduce batch size in `config.yaml` (e.g., `batch_size: 4`)
- Use smaller model (e.g., `efficientnet_b0`)
- Enable gradient checkpointing

### Issue: Face Detection Fails
**Solution:**
- Check if faces are visible in images
- Try different face detector in config: `mtcnn`, `retinaface`, or `opencv`
- Disable face detection temporarily for testing

### Issue: No Data Found
**Solution:**
- Verify data directory structure matches expected format
- Check file extensions are supported (.mp4, .avi, .mov, .mkv)
- Ensure `real/` and `fake/` subdirectories exist

### Issue: Poor Performance
**Solution:**
- Check data quality and balance (equal real/fake samples)
- Try different model architectures
- Adjust learning rate (try 1e-5 or 5e-5)
- Increase training epochs
- Add more data augmentation

## Next Steps

1. **Experiment with different models:**
   - Try ViT: `model.name: "vit"`
   - Try Ensemble: `model.name: "ensemble"`
   - Try Temporal: `model.name: "temporal"`

2. **Hyperparameter tuning:**
   - Learning rate: 1e-5 to 1e-3
   - Batch size: 4, 8, 16, 32
   - Different optimizers: AdamW, SGD

3. **Ablation studies:**
   - Compare CNN vs ViT vs Ensemble
   - Test with/without temporal modeling
   - Evaluate impact of augmentation

4. **Robustness testing:**
   - Enable in config: `evaluation.robustness_tests`
   - Test on compressed/resized/blurred images

## Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt
python setup_data_dirs.py

# 2. Add your data to data/train/, data/val/, data/test/

# 3. Train CNN baseline
python train.py --config config.yaml

# 4. Evaluate
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth

# 5. Try ensemble model
# Edit config.yaml: model.name = "ensemble"
python train.py --config config.yaml

# 6. Compare results
```

## Getting Help

- Check `README.md` for detailed documentation
- Review `config.yaml` for all configuration options
- Check training logs in `checkpoints/training_history.json`

