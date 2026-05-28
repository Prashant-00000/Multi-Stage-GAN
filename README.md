# Progressive GAN Framework

A comprehensive multi-stage GAN implementation showcasing the evolution from basic GANs to advanced architectures like CycleGAN. Each stage builds upon previous concepts with increasingly sophisticated architectures and training techniques.

## Project Overview

This project implements three progressive stages of GAN architectures:

### Stage 1: Basic GAN
- **Architecture**: Fully-connected generator and discriminator
- **Dataset**: MNIST
- **Key Features**: Label smoothing, Adam optimizer with custom betas
- **Evaluation**: FID (Fréchet Inception Distance)

### Stage 2: DCGAN (Deep Convolutional GAN)
- **Architecture**: Convolutional layers with batch normalization
- **Dataset**: MNIST
- **Key Features**: Spectral normalization-ready architecture
- **Improvements**: Better stability through convolutional design

### Stage 2b: Conditional GAN
- **Architecture**: Class-conditional generation with embedding layer
- **Dataset**: MNIST with class labels
- **Key Features**: Class embedding concatenation, per-class generation
- **Use Case**: Controlled generation of specific digit classes

### Stage 3: CycleGAN
- **Architecture**: ResNet-based generators with PatchGAN discriminators
- **Dataset**: Unpaired image-to-image translation
- **Key Features**: Cycle consistency loss, replay buffer for stability
- **Use Case**: Style transfer and domain adaptation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Stage 1 - Basic GAN
```bash
# Train
python -m src.stage1_gan.train

# Evaluate with FID
python evaluate_stage1.py
```

### Stage 2 - DCGAN
```bash
# Train
python -m src.stage2_dcgan.train
```

### Stage 2b - Conditional GAN
```bash
# Train
python -m src.stage2b_cgan.train
```

### Stage 3 - CycleGAN
```bash
# Train
python -m src.stage3_cyclegan.train
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Project Structure

```
GAN-Project/
├── src/
│   ├── common/                 # Shared utilities
│   │   ├── dataloader.py      # Data loading utils
│   │   ├── losses.py          # Loss functions
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── utils.py           # Helper functions
│   ├── stage1_gan/            # Basic GAN
│   ├── stage2_dcgan/          # Convolutional GAN
│   ├── stage2b_cgan/          # Conditional GAN
│   ├── stage3_cyclegan/       # CycleGAN
│   └── training_engine/       # Training utilities
├── evaluation/                # Evaluation metrics
├── notebooks/                 # Jupyter demos
├── outputs/                   # Checkpoints, images, logs
├── config/                    # Configuration files
└── requirements.txt
```

## Architecture Details

### Stage 1: Generator
- Input: Noise vector (z_dim=100)
- Linear → ReLU → Linear → ReLU → Linear → Tanh
- Output: Flattened MNIST image (1×784)

### Stage 1: Discriminator
- Input: Flattened image (1×784)
- Linear → LeakyReLU → Linear → LeakyReLU → Linear → Sigmoid
- Output: Binary classification (real/fake)

### Stage 2 (DCGAN): Generator
- Input: Noise vector (z_dim=100)
- Transposed convolutions with batch normalization
- Progressive upsampling to 28×28 image

### Stage 2 (DCGAN): Discriminator
- Input: 28×28 image
- Convolutional layers with batch normalization
- Binary classification output

### Stage 2b (CGAN): Additional Components
- Embedding layer for class labels
- Concatenation of noise and class embeddings

### Stage 3 (CycleGAN): Generators
- ResNet-based with skip connections
- 9 residual blocks for style transfer
- Handles arbitrary image sizes

### Stage 3 (CycleGAN): Discriminators
- PatchGAN discriminator
- Classifies image patches instead of whole image
- Better for texture and local detail preservation

## Training Techniques

### Stage 1 & 2
- Label smoothing (0.9 for real, 0.1 for fake)
- Separate discriminator and generator training steps
- Adam optimizer (lr=0.0002, betas=(0.5, 0.999))
- Gradient detachment for stability

### Stage 2b
- Conditional generation with class embeddings
- Per-class training batches

### Stage 3
- Cycle consistency loss (L1 + adversarial)
- Identity loss for color preservation
- Replay buffer for discriminator stability
- Unpaired training data support

## Evaluation Metrics

- **FID (Fréchet Inception Distance)**: Measures similarity between real and generated distributions
- **Inception Score**: Overall image quality
- **L1/L2 Distance**: Pixel-level similarity (Stage 3)

## Configuration

Edit `config/config.yaml` to customize:
- Model hyperparameters (learning rate, batch size, z_dim)
- Training settings (epochs, device)
- Data paths and dataset selection
- Evaluation metrics

## Outputs

Training generates:
- **Checkpoints**: Saved model weights
- **Images**: Sample generations during training
- **Logs**: TensorBoard logs with loss curves

## Key Concepts

### Label Smoothing
Prevents discriminator from being too confident, improving generator training stability.

### Batch Normalization
Stabilizes training by normalizing layer inputs. (Stages 2+)

### Spectral Normalization
Constrains discriminator weights for better training dynamics.

### Cycle Consistency Loss
In CycleGAN: A→B→A should reconstruct the original image, enforcing content preservation.

### PatchGAN Discriminator
Classifies overlapping patches instead of whole images, focusing on local patterns and style.

### Replay Buffer
Maintains history of generated images for discriminator training, breaking correlations in sequential batches.

## References

- Goodfellow et al., "Generative Adversarial Networks" (2014)
- Radford et al., "Unsupervised Representation Learning with DCGANs" (2015)
- Mirza & Osindero, "Conditional GANs" (2014)
- Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)

## Author Notes

This project demonstrates:
- Progressive complexity in GAN architectures
- Stability improvements through architectural choices
- Evaluation methodologies (FID, Inception Score)
- Advanced techniques (cycle consistency, PatchGAN, replay buffer)
- Clean, modular code structure for research and development

## License

MIT License
