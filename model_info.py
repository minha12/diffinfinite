import yaml
import torch
from dm import Unet, GaussianDiffusion, Trainer
from dm_masks import Unet as MaskUnet
from dm_masks import GaussianDiffusion as MaskGD
from dm_masks import Trainer as MaskTrainer

def load_model(config_file, milestone, model_type='mask'):
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create namespace for config
    config_dict = {}
    for key in config.keys():
        config_dict.update(config[key])
    
    # Initialize appropriate model based on type
    if model_type == 'mask':
        unet = MaskUnet(
            dim=config_dict['dim'],
            num_classes=config_dict['num_classes'],
            dim_mults=config_dict['dim_mults'],
            channels=config_dict['channels'],
            resnet_block_groups=config_dict['resnet_block_groups'],
            block_per_layer=config_dict['block_per_layer'],
        )
        model = MaskGD(
            unet,
            image_size=config_dict['mask_size']//8,
            timesteps=config_dict['timesteps'],
            sampling_timesteps=config_dict['sampling_timesteps'],
            loss_type='l2'
        )
        trainer = MaskTrainer(
            model,
            train_batch_size=config_dict['batch_size'],
            train_lr=config_dict['lr'],
            train_num_steps=config_dict['train_num_steps'],
            save_and_sample_every=config_dict['save_sample_every'],
            gradient_accumulate_every=config_dict['gradient_accumulate_every'],
            save_loss_every=config_dict['save_loss_every'],
            num_samples=config_dict['num_samples'],
            num_workers=config_dict['num_workers'],
            results_folder=config_dict['results_folder']
        )
    else:
        unet = Unet(
            dim=config_dict['dim'],
            num_classes=config_dict['num_classes'],
            dim_mults=config_dict['dim_mults'],
            channels=config_dict['channels'],
            resnet_block_groups=config_dict['resnet_block_groups'],
            block_per_layer=config_dict['block_per_layer'],
        )
        model = GaussianDiffusion(
            unet,
            image_size=config_dict['image_size']//8,
            timesteps=config_dict['timesteps'],
            sampling_timesteps=config_dict['sampling_timesteps'],
            loss_type='l2'
        )
        trainer = Trainer(
            model,
            train_batch_size=config_dict['batch_size'],
            train_lr=config_dict['lr'],
            train_num_steps=config_dict['train_num_steps'],
            save_and_sample_every=config_dict['save_sample_every'],
            gradient_accumulate_every=config_dict['gradient_accumulate_every'],
            save_loss_every=config_dict['save_loss_every'],
            num_samples=config_dict['num_samples'],
            num_workers=config_dict['num_workers'],
            results_folder=config_dict['results_folder']
        )
    
    trainer.load(milestone)
    trainer.ema.cuda()
    trainer.ema = trainer.ema.eval()
    
    return trainer, config_dict

def print_model_info(trainer, config_dict, model_type):
    print(f"\n{'='*50}")
    print(f"{model_type.upper()} MODEL INFORMATION")
    print(f"{'='*50}")
    
    # Print configuration parameters
    print("\nConfiguration Parameters:")
    for key, value in config_dict.items():
        print(f"{key}: {value}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(trainer.model)
    
    # Print model parameters
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    
    print("\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print device information
    print("\nDevice Information:")
    print(f"Current device: {next(trainer.model.parameters()).device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

def main():
    # Load mask model
    mask_trainer, mask_config = load_model(
        config_file='./config/mask_gen_sample5.yaml',
        milestone=5,
        model_type='mask'
    )
    print_model_info(mask_trainer, mask_config, "mask")
    
    # Load image model
    image_trainer, image_config = load_model(
        config_file='./config/image_gen_sample5.yaml',
        milestone=10,
        model_type='image'
    )
    print_model_info(image_trainer, image_config, "image")

if __name__ == '__main__':
    main()
