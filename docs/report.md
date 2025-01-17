# Training Report

## Configuration Summary

### UNet Architecture
- **Image Size**: 512x512
- **Base Dimension**: 256
- **Number of Classes**: 5
- **Dimension Multipliers**: [1, 2, 4]
- **Input Channels**: 4
- **ResNet Block Groups**: 2
- **Blocks per Layer**: 2

### Training Parameters
- **Timesteps**: 1000
- **Sampling Timesteps**: 250
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Training Steps**: 2
- **Gradient Accumulation**: Every 1 step
- **Save Sample Frequency**: Every 2 steps
- **Save Loss Frequency**: Every 2 steps
- **Number of Samples**: 4
- **Number of Workers**: 16

### Model Statistics
- **Total Parameters**: 172,101,124
- **Trainable Parameters**: 172,101,124

## Dataset Analysis

### Dataset Distribution
#### Training Set
- Class 0: 58,389 samples
- Class 1: 58,041 samples
- Class 2: 38,244 samples
- Class 3: 36,493 samples
- Class 4: 41,360 samples

#### Test Set
- Class 0: 6,488 samples
- Class 1: 6,450 samples
- Class 2: 4,250 samples
- Class 3: 4,055 samples
- Class 4: 4,596 samples

### Training Split
- **Train/Test Ratio**: 90/10

## Training Progress

### Loss Values
- Initial Loss: 1.1351
- Final Loss: 1.9007

### Image Statistics
#### Input Images
- **Format**: RGB
- **Resolution**: 512x512
- **Value Range**: [0, 255]
- **Average Mean**: ~180
- **Average Std**: ~45-50

#### Masks
- **Resolution**: 512x512
- **Classes**: 5 (0-4)
- **Format**: Single channel

## Generated Samples
- **Batch Size**: 4
- **Resolution**: 512x512
- **Channels**: 3
- **Value Range**: [0.0, 1.0]

## Technical Notes
- Training performed on CUDA-enabled device
- Model saves include state dictionary and configuration
- Warnings present regarding PyTorch's `torch.load` security implications

---
*Report generated from training output logs*