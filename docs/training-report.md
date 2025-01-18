# Training Report - DrSk 512x256 5-class Model

## Training Configuration

- Image Resolution: 512x512
- Base Dimensions: 256
- Number of Classes: 5
- Batch Size: 64
- Learning Rate: 0.0001
- Training Steps: 100,000
- Sampling Timesteps: 250/1000
- Gradient Accumulation: 2
- Latent Normalizing Scale: 1/50
- Model Architecture:
  - Channels: 4
  - Dimension Multipliers: [1, 2, 4]
  - ResNet Block Groups: 2
  - Blocks per Layer: 2

## Loss Analysis
![Training Loss Plot](loss_plot_drsk_512x256_5class_20240119_1118.png)

The loss curve shows steady convergence over training steps, with notable observations:
- Initial rapid descent in the first 1000 steps
- Stabilization period between steps 1000-2500
- Gradual refinement after step 2500

## Sample Generation Progress

### Step 1000
<div style="display: flex; justify-content: space-between;">
    <img src="../logs/drsk_512x256_5class_20240119_1118/images-10.png" width="30%">
    <img src="../logs/drsk_512x256_5class_20240119_1118/masks-10.png" width="30%">
    <img src="../logs/drsk_512x256_5class_20240119_1118/sample-10.png" width="30%">
</div>

### Step 2500
<div style="display: flex; justify-content: space-between;">
    <img src="../logs/drsk_512x256_5class_20240119_1118/images-25.png" width="30%">
    <img src="../logs/drsk_512x256_5class_20240119_1118/masks-25.png" width="30%">
    <img src="../logs/drsk_512x256_5class_20240119_1118/sample-25.png" width="30%">
</div>

### Step 5000 
<div style="display: flex; justify-content: space-between;">
    <img src="../logs/drsk_512x256_5class_20240119_1118/images-50.png" width="30%">
    <img src="../logs/drsk_512x256_5class_20240119_1118/masks-50.png" width="30%">
    <img src="../logs/drsk_512x256_5class_20240119_1118/sample-50.png" width="30%">
</div>

