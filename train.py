from fire import Fire
import yaml
from dm import Unet, GaussianDiffusion, Trainer
import json
from types import SimpleNamespace

def dict_to_namespace(d):
    json_str = json.dumps(d)
    return json.loads(json_str, object_hook=lambda d: SimpleNamespace(**d))

def main(
        config_file: str = "./config/image_gen_train.yaml",
        data_folder: str = None,
        image_size: int = 512,
        dim: int = 256,
        num_classes: int = 10,
        dim_mults: str = '1 2 4',
        channels: int = 3,
        resnet_block_groups: int = 2,
        block_per_layer: int = 2,
        timesteps: int = 1000,
        sampling_timesteps: int = 250,
        batch_size: int = 32,
        lr: float = 1e-4,
        train_num_steps: int = 250000,
        save_sample_every: int = 25000,
        save_milestone_every: int = 10000,  # Add this line
        gradient_accumulate_every: int = 1,
        save_loss_every: int = 100,
        num_samples: int = 4,
        num_workers: int = 32,
        results_folder: str = './results/run_name',
        milestone: int = None,
        extra_data_path: str = '../pathology-datasets/DRSK/full_dataset/unconditional-data',  # Add this line
):
    if config_file:
        print("Loading config from:", config_file)
        with open(config_file, 'r') as f:
            config = dict_to_namespace(yaml.safe_load(f))
    else:
        # Create config namespace from default arguments
        config = SimpleNamespace(
            unet=SimpleNamespace(
                image_size=image_size, dim=dim, num_classes=num_classes,
                dim_mults=dim_mults, channels=channels,
                resnet_block_groups=resnet_block_groups,
                block_per_layer=block_per_layer
            ),
            dm=SimpleNamespace(
                data_folder=data_folder, timesteps=timesteps,
                sampling_timesteps=sampling_timesteps, batch_size=batch_size,
                lr=lr, train_num_steps=train_num_steps,
                save_sample_every=save_sample_every,
                save_milestone_every=save_milestone_every,  # Add this line
                gradient_accumulate_every=gradient_accumulate_every,
                save_loss_every=save_loss_every, num_samples=num_samples,
                num_workers=num_workers, results_folder=results_folder,
                milestone=milestone,
                extra_data_path=extra_data_path  # Add this line
            )
        )
    debug = getattr(config.dm, 'debug', False)
    # Modified dim_mults handling to support both string and list inputs
    if isinstance(config.unet.dim_mults, str):
        dim_mults = [int(mult) for mult in config.unet.dim_mults.split(' ')]
    else:
        dim_mults = config.unet.dim_mults
    
    z_size = config.unet.image_size // 8
    
    unet = Unet(
        dim=config.unet.dim,
        num_classes=config.unet.num_classes,
        dim_mults=dim_mults,
        channels=config.unet.channels,
        resnet_block_groups=config.unet.resnet_block_groups,
        block_per_layer=config.unet.block_per_layer,
        debug=debug
    )

    
    if debug:
        print("\n=== Final Configuration [main()] ===")
        print("UNet config:")
        for k, v in vars(config.unet).items():
            print(f"  {k}: {v}")
        print("\nDiffusion config:")
        for k, v in vars(config.dm).items():
            print(f"  {k}: {v}")
        
        print("\n=== Model Summary [main()] ===")
        total_params = sum(p.numel() for p in unet.parameters())
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Input channels: {unet.channels}")
        print(f"Number of classes: {unet.num_classes}")
        print(f"Model dimension: {unet.dim}")
        print(f"Dimension multipliers: {config.unet.dim_mults}")

    model = GaussianDiffusion(
        unet,
        image_size=z_size,
        timesteps=config.dm.timesteps,
        sampling_timesteps=config.dm.sampling_timesteps,
        loss_type='l2',
        debug=debug
    )

    trainer = Trainer(
        model,
        data_folder=config.dm.data_folder,
        train_batch_size=config.dm.batch_size,
        train_lr=config.dm.lr,
        train_num_steps=config.dm.train_num_steps,
        save_and_sample_every=config.dm.save_sample_every,
        save_milestone_every=config.dm.save_milestone_every,  # Add this line
        gradient_accumulate_every=config.dm.gradient_accumulate_every,
        save_loss_every=config.dm.save_loss_every,
        num_samples=config.dm.num_samples,
        num_workers=config.dm.num_workers,
        results_folder=config.dm.results_folder,
        config_file=config_file,
        extra_data_path=config.dm.extra_data_path,
        debug=getattr(config.dm, 'debug', False)
    )

    if config.dm.milestone:
        trainer.load(config.dm.milestone)

    trainer.train()

if __name__ == '__main__':
    Fire(main)