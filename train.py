from fire import Fire
import yaml
from dm import Unet, GaussianDiffusion, Trainer

def main(
        config_file: str = None,
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
        gradient_accumulate_every: int = 1,
        save_loss_every: int = 100,
        num_samples: int = 4,
        num_workers: int = 32,
        results_folder: str = './results/run_name',
        milestone: int = None,
):
    if config_file:
        print("Loading config from:", config_file)
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        unet_conf = config.get('unet', {})
        image_size = unet_conf.get('image_size', image_size)
        dim = unet_conf.get('dim', dim)
        num_classes = unet_conf.get('num_classes', num_classes)
        dim_mults_list = unet_conf.get('dim_mults', dim_mults)
        channels = unet_conf.get('channels', channels)
        resnet_block_groups = unet_conf.get('resnet_block_groups', resnet_block_groups)
        block_per_layer = unet_conf.get('block_per_layer', block_per_layer)

        dm_conf = config.get('dm', {})
        data_folder = dm_conf.get('data_folder', data_folder)
        timesteps = dm_conf.get('timesteps', timesteps)
        sampling_timesteps = dm_conf.get('sampling_timesteps', sampling_timesteps)
        batch_size = dm_conf.get('batch_size', batch_size)
        lr = dm_conf.get('lr', lr)
        train_num_steps = dm_conf.get('train_num_steps', train_num_steps)
        save_sample_every = dm_conf.get('save_sample_every', save_sample_every)
        gradient_accumulate_every = dm_conf.get('gradient_accumulate_every', gradient_accumulate_every)
        save_loss_every = dm_conf.get('save_loss_every', save_loss_every)
        num_samples = dm_conf.get('num_samples', num_samples)
        num_workers = dm_conf.get('num_workers', num_workers)
        results_folder = dm_conf.get('results_folder', results_folder)
        milestone = dm_conf.get('milestone', milestone)

    dim_mults=[int(mult) for mult in dim_mults.split(' ')]

    z_size=image_size//8
    
    unet = Unet(
            dim=dim,
            num_classes=num_classes,
            dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups = resnet_block_groups,
            block_per_layer=block_per_layer,
        )

    model = GaussianDiffusion(
            unet,
            image_size=z_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type='l2')
    print(data_folder)
    trainer = Trainer(
            model,
            data_folder=data_folder,
            train_batch_size=batch_size,
            train_lr=lr,
            train_num_steps=train_num_steps,
            save_and_sample_every=save_sample_every,
            gradient_accumulate_every=gradient_accumulate_every,
            save_loss_every=save_loss_every,
            num_samples=num_samples,
            num_workers=num_workers,
            results_folder=results_folder)

    if milestone:
        trainer.load(milestone)
    #print(vars(trainer))    
    trainer.train()

if __name__=='__main__':
    Fire(main)