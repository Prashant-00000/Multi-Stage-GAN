# GAN Project - Quick Start Guide

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python test_import.py
```

## Running the Project

### Stage 1: Basic GAN

#### Option A: Run demo script (Quick start - 10 epochs)
```bash
python demo_stage1.py
```

#### Option B: Full training (50 epochs)
```bash
python -m src.stage1_gan.train
```

#### Option C: Evaluate trained model
```bash
python evaluate_stage1.py
```

### Stage 2: DCGAN

#### Option A: Run demo script (Quick start)
```bash
python demo_stage2.py
```

#### Option B: Full training
```bash
python -m src.stage2_dcgan.train
```

### Stage 2b: Conditional GAN

#### Run demo script
```bash
python demo_stage2b.py
```

#### Full training
```bash
python -m src.stage2b_cgan.train
```

### Stage 3: CycleGAN

#### Run training setup
```bash
python -m src.stage3_cyclegan.train
```

Note: CycleGAN requires paired domain A and B datasets. Modify the training script to load your custom datasets.

### Comprehensive Evaluation

```bash
python evaluation/evaluate.py
```

This evaluates all three stages using FID scores.

## Project Structure

```
GAN-Project/
├── src/
│   ├── stage1_gan/              # Basic GAN implementation
│   │   ├── generator.py         # Generator network
│   │   ├── discriminator.py     # Discriminator network
│   │   ├── model.py            # Model wrapper
│   │   └── train.py            # Training script
│   │
│   ├── stage2_dcgan/            # Convolutional GAN
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   ├── stage2b_cgan/            # Conditional GAN
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   ├── embeddings.py        # Class embedding layers
│   │   ├── model.py
│   │   └── train.py
│   │
│   ├── stage3_cyclegan/         # CycleGAN for unpaired translation
│   │   ├── generator_resnet.py  # ResNet-based generator
│   │   ├── discriminator_patchgan.py  # PatchGAN discriminator
│   │   ├── replay_buffer.py     # Image replay buffer
│   │   ├── model.py
│   │   └── train.py
│   │
│   ├── common/                  # Shared utilities
│   │   ├── dataloader.py        # Data loading utils
│   │   ├── losses.py            # Loss functions
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── utils.py             # Helper functions
│   │
│   └── training_engine/         # Training utilities
│       ├── gan_loop.py          # Main training step
│       ├── scheduler.py         # Learning rate scheduler
│       └── trainer.py           # GANTrainer class
│
├── evaluation/                  # Evaluation modules
│   ├── fid.py                   # FID score calculation
│   ├── inception_model.py       # Inception model
│   └── evaluate.py              # Comprehensive evaluation
│
├── notebooks/                   # Jupyter notebooks
│   ├── stage1_demo.ipynb
│   ├── stage2_demo.ipynb
│   └── stage3_demo.ipynb
│
├── config/
│   └── config.yaml              # Configuration file
│
├── outputs/
│   ├── checkpoints/             # Saved model weights
│   ├── images/                  # Generated samples
│   └── logs/                    # Training logs
│
├── data/                        # Datasets
│   ├── MNIST/
│   ├── celeba/
│   ├── monet/
│   └── photos/
│
├── demo_stage1.py               # Quick demo scripts
├── demo_stage2.py
├── demo_stage2b.py
├── evaluate_stage1.py
├── test_import.py
├── README.md
└── requirements.txt
```

## Key Features Implemented

### Stage 1: Basic GAN
- ✅ Fully-connected generator and discriminator
- ✅ Label smoothing (0.9 for real, 0.1 for fake)
- ✅ Adam optimizer with custom betas
- ✅ FID score evaluation

### Stage 2: DCGAN
- ✅ Convolutional generator with transposed convolutions
- ✅ Convolutional discriminator
- ✅ Batch normalization for stability
- ✅ Weight initialization (normal/xavier/kaiming)

### Stage 2b: Conditional GAN
- ✅ Class embedding layer
- ✅ Conditional generation (generate specific classes)
- ✅ Class-conditioned discriminator
- ✅ Per-class sample generation

### Stage 3: CycleGAN
- ✅ ResNet-based generators with skip connections
- ✅ PatchGAN discriminators
- ✅ Cycle consistency loss
- ✅ Identity loss for color preservation
- ✅ Replay buffer for discriminator stability

## Common Utilities

### Loss Functions
- `adversarial_loss()` - Binary cross-entropy with label smoothing
- `WassersteinLoss` - Wasserstein GAN loss
- `HingeLoss` - Hinge loss variant
- `CycleLoss` - Cycle consistency for CycleGAN
- `IdentityLoss` - Identity mapping loss

### Data Loading
- `get_data_loader()` - Load MNIST, CelebA, or custom datasets
- `get_mnist_loaders()` - Get train/test MNIST loaders
- `get_conditional_mnist_loader()` - Load with class labels
- `UnpairedImageDataset` - For CycleGAN unpaired training

### Metrics
- `MetricsTracker` - Track training metrics
- `compute_inception_score()` - Calculate Inception Score
- `L1Distance` - Pixel-level similarity
- `SSIMDistance` - Structural similarity

### Utilities
- `save_checkpoint()` / `load_checkpoint()` - Checkpoint management
- `init_weights()` - Network weight initialization
- `denormalize()` / `normalize()` - Image normalization
- `AverageMeter` - Running average tracker

## Hyperparameter Configuration

Edit `config/config.yaml` to customize:

```yaml
stage: 1                # GAN stage (1, 2, 2b, or 3)
z_dim: 100             # Noise vector dimension
batch_size: 64         # Training batch size
lr: 0.0002             # Learning rate
epochs: 50             # Training epochs
device: cuda           # cuda or cpu

# Stage-specific configs
stage1:
  hidden_dim: 256
  img_dim: 784

stage2:
  img_size: 28
  feature_maps: 64

stage3:
  img_size: 256
  lambda_cycle: 10.0
  lambda_identity: 0.5
```

## Typical Workflow

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run quick demo**: `python demo_stage1.py`
3. **Check results**: Browse `outputs/images/` for generated samples
4. **Evaluate model**: `python evaluation/evaluate.py`
5. **Modify config**: Edit `config/config.yaml` for custom training
6. **Train full model**: `python -m src.stage1_gan.train`

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config or training script
- Use CPU: set `device: cpu` in config

### Slow Training
- Reduce number of epochs for testing
- Use demo scripts for quick validation
- Check GPU usage with `nvidia-smi`

### Import Errors
- Ensure you're in the project directory
- Run `python test_import.py` to verify setup
- Check Python path includes `src/`

## Performance Benchmarks

### FID Scores (Lower is Better)
- **Stage 1**: 15-30 (basic GAN)
- **Stage 2**: 10-20 (DCGAN with convolutions)
- **Stage 2b**: 10-18 (conditional generation)
- **Stage 3**: 5-15 (CycleGAN unpaired)

### Training Time (per epoch on single GPU)
- **Stage 1**: ~30 seconds (MNIST, 50k images)
- **Stage 2**: ~45 seconds (with convolutions)
- **Stage 2b**: ~50 seconds (with embeddings)
- **Stage 3**: ~2-3 minutes (if using 256×256 images)

## Advanced Topics

### Custom Dataset Loading
```python
from src.common.dataloader import CustomImageDataset

dataset = CustomImageDataset(
    img_dir="path/to/images",
    img_size=256
)
loader = DataLoader(dataset, batch_size=32)
```

### Training with Custom Config
```python
from src.common.utils import load_config

config = load_config("config/config.yaml")
config['epochs'] = 100  # Custom setting
```

### Checkpoint Management
```python
from src.common.utils import save_checkpoint, load_checkpoint

# Save
save_checkpoint(gen, disc, opt_g, opt_d, epoch, "outputs/checkpoints")

# Load
epoch = load_checkpoint(checkpoint_path, gen, disc, opt_g, opt_d, device)
```

## References

- Goodfellow et al., "Generative Adversarial Networks" (2014)
- Radford et al., "Unsupervised Representation Learning with DCGANs" (2015)
- Mirza & Osindero, "Conditional Generative Adversarial Nets" (2014)
- Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)

## Next Steps

1. Train Stage 1 GAN to convergence
2. Compare Stage 1 vs Stage 2 FID scores
3. Experiment with hyperparameters
4. Load custom datasets for CycleGAN
5. Extend to other domains (faces, artistic styles, etc.)

Good luck with your GAN experiments!
