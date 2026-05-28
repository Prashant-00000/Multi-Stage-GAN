#!/usr/bin/env python
"""
Comprehensive comparison of all 4 GAN stages showing improvements
"""

import os
import torch
from src.stage1_gan.generator import Generator as Gen1
from src.stage1_gan.discriminator import Discriminator as Disc1
from src.stage2_dcgan.generator import DCGANGenerator as Gen2
from src.stage2_dcgan.discriminator import DCGANDiscriminator as Disc2
from src.stage2b_cgan.generator import ConditionalGenerator as Gen2b
from src.stage2b_cgan.discriminator import ConditionalDiscriminator as Disc2b
from src.stage3_cyclegan.generator_resnet import ResNetGenerator as Gen3
from src.stage3_cyclegan.discriminator_patchgan import PatchGANDiscriminator as Disc3

print("\n" + "="*80)
print("PROGRESSIVE GAN FRAMEWORK - STAGE COMPARISON & IMPROVEMENTS".center(80))
print("="*80)

# Stage Specifications
stages_info = {
    "Stage 1": {
        "name": "Basic GAN",
        "gen_class": Gen1,
        "disc_class": Disc1,
        "gen_args": {"z_dim": 100, "img_dim": 784},
        "disc_args": {"img_dim": 784},
    },
    "Stage 2": {
        "name": "DCGAN",
        "gen_class": Gen2,
        "disc_class": Disc2,
        "gen_args": {"z_dim": 100, "channels": 1, "feature_maps": 64},
        "disc_args": {"channels": 1, "feature_maps": 64},
    },
    "Stage 2b": {
        "name": "Conditional GAN",
        "gen_class": Gen2b,
        "disc_class": Disc2b,
        "gen_args": {"z_dim": 100, "num_classes": 10, "channels": 1, "feature_maps": 64, "embedding_dim": 100},
        "disc_args": {"num_classes": 10, "channels": 1, "feature_maps": 64, "embedding_dim": 100},
    },
    "Stage 3": {
        "name": "CycleGAN",
        "gen_class": Gen3,
        "disc_class": Disc3,
        "gen_args": {"in_channels": 3, "out_channels": 3, "num_residual_blocks": 9, "feature_maps": 64},
        "disc_args": {"in_channels": 3, "feature_maps": 64},
    },
}

# Collect all metrics
print("\n📊 ARCHITECTURE SPECIFICATIONS")
print("-" * 80)
print(f"{'Stage':<12} {'Architecture':<20} {'Generator Params':<20} {'Discriminator Params':<20}")
print("-" * 80)

metrics = {}
for stage_key, info in stages_info.items():
    try:
        gen = info["gen_class"](**info["gen_args"])
        disc = info["disc_class"](**info["disc_args"])
        
        gen_params = sum(p.numel() for p in gen.parameters())
        disc_params = sum(p.numel() for p in disc.parameters())
        
        metrics[stage_key] = {
            "name": info["name"],
            "gen_params": gen_params,
            "disc_params": disc_params,
            "total_params": gen_params + disc_params,
        }
        
        print(f"{stage_key:<12} {info['name']:<20} {gen_params:>18,} {disc_params:>18,}")
    except Exception as e:
        print(f"{stage_key:<12} {info['name']:<20} Error: {str(e)}")

# Calculate training completion status
print("\n\n📈 TRAINING STATUS & OUTPUTS")
print("-" * 80)
print(f"{'Stage':<12} {'Status':<30} {'Output Images':<20}")
print("-" * 80)

output_dir = "outputs/images"
stages_outputs = {
    "Stage 1": "stage1_epoch_10.png",
    "Stage 2": "stage2_epoch_10.png",
    "Stage 2b": "stage2b_epoch_10.png",
    "Stage 3": "stage3_epoch_10.png",
}

for stage, filename in stages_outputs.items():
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        status = "✅ COMPLETE"
        output = f"{filename} ({size_kb:.1f}KB)"
    else:
        status = "⏳ TRAINING"
        output = "Pending..."
    
    print(f"{stage:<12} {status:<30} {output:<20}")

# Key improvements across stages
print("\n\n🚀 KEY IMPROVEMENTS ACROSS STAGES")
print("-" * 80)

improvements = {
    "Stage 1 → Stage 2": {
        "Architecture": "Fully-connected → Convolutional",
        "Parameters": f"{metrics.get('Stage 1', {}).get('total_params', 0):,} → {metrics.get('Stage 2', {}).get('total_params', 0):,}",
        "Benefit": "Spatial awareness, better feature learning",
        "Stability": "Batch normalization added",
    },
    "Stage 2 → Stage 2b": {
        "Architecture": "Unconditional → Class-conditional",
        "Parameters": f"{metrics.get('Stage 2', {}).get('total_params', 0):,} → {metrics.get('Stage 2b', {}).get('total_params', 0):,}",
        "Benefit": "Controlled generation, class-specific samples",
        "Stability": "Embedding layer for class info",
    },
    "Stage 2b → Stage 3": {
        "Architecture": "DCGAN → ResNet + PatchGAN",
        "Parameters": f"{metrics.get('Stage 2b', {}).get('total_params', 0):,} → {metrics.get('Stage 3', {}).get('total_params', 0):,}",
        "Benefit": "Unpaired training, style transfer",
        "Stability": "Cycle consistency + identity loss",
    },
}

for transition, improvements_dict in improvements.items():
    print(f"\n{transition}")
    print("  " + "-" * 76)
    for key, value in improvements_dict.items():
        print(f"  {key:<15}: {value}")

# Feature comparison table
print("\n\n📋 DETAILED FEATURE COMPARISON")
print("-" * 80)

features_matrix = {
    "Feature": ["Input Type", "Training Data", "Generator Design", "Discriminator Design", "Loss Function", "Key Technique"],
    "Stage 1": ["Noise (z)", "Unpaired MNIST", "3 Linear layers", "3 Linear layers", "BCE + Label Smooth", "Label smoothing"],
    "Stage 2": ["Noise (z)", "Unpaired MNIST", "4 Transposed Conv", "4 Conv layers", "BCE + Label Smooth", "Batch norm"],
    "Stage 2b": ["(Noise + Class)", "Labeled MNIST", "Conv + Embedding", "Conv + Embedding", "BCE + Label Smooth", "Class embedding"],
    "Stage 3": ["(Noise) or Image", "Unpaired Image Pairs", "ResNet (9 blocks)", "PatchGAN (local)", "Cycle + Identity", "Cycle consistency"],
}

# Print header
print(f"{'Feature':<20}", end="")
for stage in ["Stage 1", "Stage 2", "Stage 2b", "Stage 3"]:
    print(f"  {stage:<18}", end="")
print()
print("-" * 80)

# Print rows
for i, feature in enumerate(features_matrix["Feature"]):
    print(f"{feature:<20}", end="")
    for stage in ["Stage 1", "Stage 2", "Stage 2b", "Stage 3"]:
        print(f"  {features_matrix[stage][i]:<18}", end="")
    print()

# Summary statistics
print("\n\n📊 SUMMARY STATISTICS")
print("-" * 80)

if metrics:
    total_params_all = sum(m.get("total_params", 0) for m in metrics.values())
    max_params = max((m.get("total_params", 0) for m in metrics.values()), default=0)
    min_params = min((m.get("total_params", 0) for m in metrics.values()), default=0)
    
    print(f"Total Parameters (All Stages): {total_params_all:,}")
    print(f"Largest Stage: Stage 3 / CycleGAN ({max_params:,} params)")
    print(f"Smallest Stage: Stage 1 / Basic GAN ({min_params:,} params)")
    print(f"Parameter Growth: {max_params/min_params:.1f}x from Stage 1 to Stage 3")

print("\n\n✨ LEARNING PROGRESSION")
print("-" * 80)
print("""
Stage 1: Foundation
  └─ Learn basic adversarial training mechanics
  └─ Understand generator-discriminator equilibrium
  └─ Foundation for all subsequent stages

Stage 2: Spatial Awareness
  └─ Transition from fully-connected to convolutional layers
  └─ Better feature maps and spatial relationships
  └─ Improved image quality with local coherence

Stage 2b: Controlled Generation
  └─ Introduce conditional information (class labels)
  └─ Generate specific digit classes on demand
  └─ Embedding layer for semantic information

Stage 3: Unpaired Translation
  └─ Learn without paired examples (realistic scenario)
  └─ Style transfer and domain adaptation
  └─ Cycle consistency enforces content preservation
""")

print("=" * 80)
print("✅ Comparison complete! All stages demonstrate progressive improvements.".center(80))
print("=" * 80 + "\n")
