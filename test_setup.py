"""Test script to verify project setup and basic functionality."""
import sys
from pathlib import Path

def test_imports():
    """Test if all imports work."""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
        
        import timm
        print(f"✓ timm")
        
        import transformers
        print(f"✓ transformers")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import albumentations
        print(f"✓ albumentations")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import sklearn
        print(f"✓ scikit-learn")
        
        # Test project imports
        from src.utils.config import load_config
        print("✓ src.utils.config")
        
        from src.models import CNNModel, ViTDeepFakeModel, TemporalModel, EnsembleModel
        print("✓ src.models")
        
        from src.data import DeepFakeDataset
        print("✓ src.data")
        
        from src.training import Trainer, FocalLoss
        print("✓ src.training")
        
        from src.evaluation import Evaluator
        print("✓ src.evaluation")
        
        from src.explainability import GradCAMExplainer, AttentionVisualizer
        print("✓ src.explainability")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

def test_config():
    """Test config loading."""
    print("\nTesting config loading...")
    try:
        from src.utils.config import load_config
        config = load_config("config.yaml")
        print("✓ Config loaded successfully")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
        print(f"  - Learning rate: {config['training']['learning_rate']}")
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        import torch
        from src.models import CNNModel
        
        # Test CNN model
        model = CNNModel(
            backbone="efficientnet_b0",
            num_classes=1,
            dropout=0.5,
            use_pretrained=False,  # Don't download for test
            hidden_dim=512
        )
        print("✓ CNN model created")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ CNN forward pass successful (output shape: {output.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structure():
    """Test data directory structure."""
    print("\nTesting data structure...")
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            print("⚠ Data directory doesn't exist. Run: python setup_data_dirs.py")
            return False
        
        required_dirs = [
            "train/real", "train/fake",
            "val/real", "val/fake",
            "test/real", "test/fake"
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            full_path = data_dir / dir_path
            if full_path.exists():
                print(f"✓ {dir_path}")
            else:
                print(f"⚠ {dir_path} - missing (create with: python setup_data_dirs.py)")
                all_exist = False
        
        return all_exist
    except Exception as e:
        print(f"❌ Data structure test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("DeepFake Detection System - Setup Test")
    print("=" * 50)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test config
    results.append(("Config", test_config()))
    
    # Test model creation
    results.append(("Model Creation", test_model_creation()))
    
    # Test data structure
    results.append(("Data Structure", test_data_structure()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Run: python setup_data_dirs.py")
        print("2. Add your data to data/train/, data/val/, data/test/")
        print("3. Run: python train.py --config config.yaml")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Create data directories: python setup_data_dirs.py")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

