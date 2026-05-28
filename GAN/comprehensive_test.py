"""
Comprehensive Import Test
Verifies all project modules are properly importable
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all major imports"""
    print("="*70)
    print("GAN Project - Comprehensive Import Test")
    print("="*70)
    
    tests = {
        "Core Dependencies": [
            ("torch", "import torch"),
            ("torchvision", "import torchvision"),
            ("numpy", "import numpy"),
            ("matplotlib", "import matplotlib.pyplot"),
        ],
        
        "Stage 1: Basic GAN": [
            ("Generator", "from src.stage1_gan.generator import Generator"),
            ("Discriminator", "from src.stage1_gan.discriminator import Discriminator"),
            ("Stage1 Model", "from src.stage1_gan.model import GAN"),
        ],
        
        "Stage 2: DCGAN": [
            ("DCGAN Generator", "from src.stage2_dcgan.generator import DCGANGenerator"),
            ("DCGAN Discriminator", "from src.stage2_dcgan.discriminator import DCGANDiscriminator"),
            ("Stage2 Model", "from src.stage2_dcgan.model import DCGAN"),
        ],
        
        "Stage 2b: Conditional GAN": [
            ("Conditional Generator", "from src.stage2b_cgan.generator import ConditionalGenerator"),
            ("Conditional Discriminator", "from src.stage2b_cgan.discriminator import ConditionalDiscriminator"),
            ("Class Embedding", "from src.stage2b_cgan.embeddings import ClassEmbedding"),
            ("Stage2b Model", "from src.stage2b_cgan.model import CGAN"),
        ],
        
        "Stage 3: CycleGAN": [
            ("ResNet Generator", "from src.stage3_cyclegan.generator_resnet import ResNetGenerator"),
            ("PatchGAN Discriminator", "from src.stage3_cyclegan.discriminator_patchgan import PatchGANDiscriminator"),
            ("Replay Buffer", "from src.stage3_cyclegan.replay_buffer import ReplayBufferFixed"),
            ("Stage3 Model", "from src.stage3_cyclegan.model import CycleGAN"),
        ],
        
        "Training Engine": [
            ("Training Step", "from src.training_engine.gan_loop import train_step"),
            ("Scheduler", "from src.training_engine.scheduler import Scheduler"),
            ("GAN Trainer", "from src.training_engine.trainer import GANTrainer"),
        ],
        
        "Common Utilities": [
            ("Data Loading", "from src.common.dataloader import get_data_loader, get_mnist_loaders"),
            ("Losses", "from src.common.losses import adversarial_loss, CycleLoss, IdentityLoss"),
            ("Metrics", "from src.common.metrics import MetricsTracker, compute_inception_score"),
            ("Utils", "from src.common.utils import save_checkpoint, load_checkpoint, init_weights"),
        ],
        
        "Evaluation": [
            ("FID Calculator", "from evaluation.fid import FIDCalculator"),
            ("Inception Model", "from evaluation.inception_model import InceptionModel"),
            ("GAN Evaluator", "from evaluation.evaluate import GANEvaluator"),
        ],
    }
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    for category, imports in tests.items():
        print(f"\n{category}:")
        print("-" * 70)
        results[category] = []
        
        for name, import_stmt in imports:
            total_tests += 1
            try:
                exec(import_stmt)
                print(f"  ✅ {name:.<50} PASS")
                results[category].append((name, True))
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {name:.<50} FAIL")
                print(f"     Error: {str(e)[:60]}")
                results[category].append((name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed ({100*passed_tests//total_tests}%)")
    print(f"{'='*70}")
    
    if passed_tests == total_tests:
        print("\n🎉 All imports successful! Project is ready to use.\n")
        print("Next steps:")
        print("  1. Run: python demo_stage1.py")
        print("  2. Run: python demo_stage2.py")
        print("  3. Run: python demo_stage2b.py")
        print("  4. Run: python evaluation/evaluate.py")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} import(s) failed. Check errors above.\n")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
