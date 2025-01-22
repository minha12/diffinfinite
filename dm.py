import math
import copy
import os
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, lr_scheduler
from torchvision import transforms as T, utils

from PIL import Image
from ema_pytorch import EMA

from accelerate import Accelerator

from dataset import import_dataset, ComposeState, RandomRotate90
from utils.helpers import *
from utils.modules import *

from diffusers import DiffusionPipeline

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

# Constants


torch.set_float32_matmul_precision('high')
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def normalize_to_neg_one_to_one(x):
    return x * 2.0 - 1.0

# Model


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        block_per_layer=2,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        debug = False,  # Add debug parameter
    ):
        super().__init__()

        # Store instance attributes
        self.num_classes = num_classes
        self.dim = dim
        self.channels = channels
        self.debug = debug
        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            blocks=[]
            for i in range(block_per_layer):
                blocks+=[block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),]
            
            blocks+=[Residual(PreNorm(dim_in, CrossAttention(dim_in))),]
            blocks+=[Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),]
            
            self.downs.append(nn.ModuleList(blocks))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, CrossAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            blocks=[]
            for i in range(block_per_layer):
                blocks+=[block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),]
            
            blocks+=[Residual(PreNorm(dim_out, CrossAttention(dim_out))),]
            blocks+=[Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)]
            
            self.ups.append(nn.ModuleList(blocks))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # Add this line to store num_classes as instance attribute
        self.num_classes = num_classes

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits
        
        # condition zero classifier free
        args = tuple(arg if i!=2 else torch.zeros_like(arg, device=arg.device).int() for i,arg in enumerate(args))
        null_logits = self.forward(*args, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, classes):
        batch, device = x.shape[0], x.device
        
        # Keep original masks for attention layers
        masks = classes.clone()
        
        # Calculate class distribution
        class_dist = torch.zeros((batch, self.num_classes), device=device)
        for c in range(self.num_classes):
            class_mask = (classes == c).float()
            class_dist[:, c] = class_mask.mean(dim=(1,2,3))
        
        # Get embeddings using distribution
        classes_emb = self.classes_emb(torch.arange(self.num_classes, device=device))  # [num_classes, dim]
        weighted_emb = (classes_emb[None, :, :] * class_dist[:, :, None]).sum(1)  # [batch, dim]
        
        # Process through existing MLP
        c = self.classes_mlp(weighted_emb)

        # Add debug prints
        if self.debug:
            print(f"Size of masks: {masks.size()}")
            print(f"Size of class_dist: {class_dist.size()}")
            print((f"Value of classes distribution: {class_dist}" ))
            print(f"Size of weighted_emb: {weighted_emb.size()}")
            print(f"Size of classes_emb: {classes_emb.size()}")
            print(f"Size of c: {c.size()}")
            print(f"Value of c: {c}")
        
        # Continue with rest of forward pass...
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []

        for *blocks, attn, downsample in self.downs:
            for i, block in enumerate(blocks):
                x = block(x, t, c)
                if i < len(blocks)-1:
                    h.append(x)
                
            x = attn(x, masks)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x, masks)
        x = self.mid_block2(x, t, c)

        for *blocks, attn, upsample in self.ups:
            for block in blocks:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block(x, t, c)
            x = attn(x, masks)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        debug = False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        self.debug = debug

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale = 3., clip_x_start = False):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, cond_scale)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, 
                                                                                  x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 3., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 3., verbose=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        if verbose:
            print('Start Sampling\n')
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))
            
        for t in iterator:
            img, x_start = self.p_sample(img, t, classes, cond_scale)
        return img

    
    @torch.no_grad()
    def ddim_onemask(self, x_t, labels, masks, time, time_next, cond_scale):
        
        masks=torch.cat([(mask*labels[i])[None] for i,mask in enumerate(masks)],0).to(x_t.device)
        cond_scale = 1.0 if (labels == 0).all().item() else cond_scale
        time_cond = torch.full((x_t.shape[0],), time, device=x_t.device, dtype=torch.long)
        
        pred_noise, x_start, *_ = self.model_predictions(x_t, time_cond, masks, 
                                                         cond_scale = cond_scale, 
                                                         clip_x_start = True)
        
        if time_next > 0:
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t, device=x_t.device)

            x_next = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        else:
            x_next = x_start

        return x_next 
    
    @torch.no_grad()
    def ddim_multimask(self, x_t, masks, time, time_next, cond_scale=3.0):
        
        
        labels=[torch.unique(mask) for mask in masks]
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1).int()
        
        x_next = torch.zeros_like(x_t, device=x_t.device)
        
        for i in range(len(padded_labels[0])):
            labels=padded_labels[:,i]
            indices = torch.where(labels != -1)[0]
            sub_images, sub_masks, sub_labels=map(lambda x: x[indices].clone(), (x_t,masks,labels)) 
            #exclude other labels from the sub_masks
            sub_masks = (sub_masks == sub_labels[:, None, None, None]).float() 
            
            x_next[indices] += self.ddim_onemask(sub_images, sub_labels, sub_masks, time, 
                                        time_next, cond_scale=cond_scale)*sub_masks      
            
        return x_next

    @torch.no_grad()
    def ddim_sample(self, images, classes, shape, cond_scale = 3., 
                    sampling_timesteps=None,
                    clip_denoised=True, 
                    inp_mask=None, verbose=False):
        
        sampling_timesteps = self.sampling_timesteps if not sampling_timesteps else sampling_timesteps
        
        # Prepare (time, time_next) pairs
        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 

        # Initialise step t=T
        x_t = torch.randn(shape, device = images.device)
        
        # Downsample masks for latent space
        masks=classes.clone().float()
        vmin,vmax=masks.min(),masks.max()
        masks=T.Lambda(lambda x: F.interpolate(x,size=self.image_size))(masks)
        masks=torch.clamp(torch.round(masks),vmin,vmax)

        if verbose:
            print('Start Sampling\n')
            iterator = tqdm(time_pairs, desc = 'sampling loop time step')
        else:
            iterator = time_pairs
            
        for time, time_next in iterator:  
            x_t = self.ddim_multimask(x_t, masks, time, time_next, cond_scale=cond_scale)
            
            noise = torch.randn(x_t.shape, device=x_t.device)
            time_cond = torch.full((x_t.shape[0],), time, device=x_t.device, dtype=torch.long)
            if time_next>0:
                x_0_noised = self.q_sample(x_start = images.clone(), 
                                           t = time_cond, noise = noise)
                if inp_mask is not None:
                    x_t = x_0_noised*(1-inp_mask) + x_t*inp_mask
    
        return x_t

    @torch.no_grad()
    def sample(self, images, classes, inp_mask=None, sampling_timesteps=250, cond_scale = 3., verbose=False):
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(images=images, classes=classes, inp_mask=inp_mask, shape=(batch_size, channels, image_size, image_size),  cond_scale=cond_scale, sampling_timesteps=sampling_timesteps, clip_denoised=True, verbose=verbose)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, *, classes, noise = None):
        if self.debug:
            print("\n=== Diffusion Step Info [GaussianDiffusion.p_losses()] ===")
            print(f"Input shape: {x_start.shape}")
            print(f"Input dtype: {x_start.dtype}")
            print(f"Timestep: {t}")
            
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        
        # predict and take gradient step

        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        model_out=torch.nan_to_num(model_out)
        target=torch.nan_to_num(target)
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        if self.debug:
            print(f"=== Model Predictions [GaussianDiffusion.p_losses()] ===")
            print(f"Predicted noise shape: {model_out.shape}")
            print(f"Target noise shape: {target.shape}")
            
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

class Trainer:
    def __init__(
        self,
        diffusion_model,
        *,
        train_batch_size = 16,
        cond_scale = 3.0,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        save_milestone_every = 10000,  # Add this line
        save_loss_every = 100,
        num_workers = 0,
        num_samples = 4,
        data_folder = None,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        out_size=None,
        config_file = None,
        extra_data_path = None,
        norm_scale = 0.18215, # Standard SD scaling 0.18215 or 1/50
        debug = False,  # Add debug parameter
    ):
        super().__init__()

        # Store debug flag as instance attribute
        self.debug = debug

        # 1. Initialize accelerator before model creation
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            gradient_accumulation_steps=gradient_accumulate_every
        )
        # Get device from accelerator
        self.device = self.accelerator.device

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_milestone_every = save_milestone_every  # Add this line
        self.save_loss_every = save_loss_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.cond_scale = cond_scale
        self.norm_scale = norm_scale
        if data_folder:
            transform=ComposeState([
                        T.ToTensor(),
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                        RandomRotate90(),
                        ])

            train_loader, test_loader = import_dataset(data_folder,
                                                batch_size=train_batch_size,   
                                                transform=transform,
                                                config_file=config_file,
                                                extra_data_path=extra_data_path,
                                                debug=debug)  # Pass debug flag

            train_loader, test_loader = self.accelerator.prepare(train_loader,test_loader)
            self.dl = cycle(train_loader)
            self.test_loader= cycle(test_loader)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        self.running_loss=[]
        self.running_lr=[]
        self.learning_rate = train_lr  # Store learning rate

        # 2. Create VAE before DDP wrapping
        self.vae = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base"
        ).vae

        # 3. Prepare models separately
        # self.model = self.accelerator.prepare(self.model)
        self.vae = self.accelerator.prepare(self.vae)

        self.scheduler = lr_scheduler.OneCycleLR(self.opt, max_lr=train_lr, total_steps=train_num_steps)
        self.model, self.opt, self.ema, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.ema, self.scheduler)

        # Add TensorBoard writer
        self.writer = SummaryWriter(os.path.join(self.results_folder, 'tensorboard'))

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'loss': self.running_loss,
            'lr': self.running_lr,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.accelerator.get_state_dict(self.ema),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

        # Close TensorBoard writer before saving
        if self.accelerator.is_local_main_process:
            self.writer.flush()

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.accelerator.device)
        
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.ema = self.accelerator.unwrap_model(self.ema)
        self.ema.load_state_dict(data['ema'])
        self.running_loss = data['loss']
        self.running_lr = data['lr']

        if exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        # Access base scheduler or use stored max_lr
        base_scheduler = getattr(self.scheduler, 'scheduler', self.scheduler)
        max_lr = getattr(base_scheduler, 'max_lrs', [self.learning_rate])[0]
        
        # Create new scheduler with remaining steps
        remaining_steps = self.train_num_steps - self.step
        self.scheduler = lr_scheduler.OneCycleLR(self.opt, 
                                               max_lr=self.learning_rate,  # Use stored value
                                               total_steps=remaining_steps)

        self.model, self.opt, self.ema, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.ema, self.scheduler)
            
    def train_loop(self, imgs, masks):
        if self.debug:
            print("\n=== Training Loop Image Details [Trainer.train_loop()] ===")
            print(f"Input images metadata:")
            print(f"  Shape: {imgs.shape}")
            print(f"  Dtype: {imgs.dtype}")
            print(f"  Device: {imgs.device}")
            print(f"  Value range: [{imgs.min():.2f}, {imgs.max():.2f}]")
            print(f"  Mean: {imgs.mean():.2f}")
            print(f"  Std: {imgs.std():.2f}")
            print(f"\nInput masks metadata:")
            print(f"  Shape: {masks.shape}")
            print(f"  Dtype: {masks.dtype}")
            print(f"  Device: {masks.device}")
            print(f"  Unique values: {torch.unique(masks).tolist()}")
            print(f"  Value counts:")
            unique, counts = torch.unique(masks, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"    {val}: {count}")

        # Move VAE encoding outside of the training loop
        with torch.no_grad():
            vae = self.accelerator.unwrap_model(self.vae)
            imgs = vae.encode(imgs).latent_dist.sample() * self.norm_scale #correct scale

        with self.accelerator.autocast():
            loss = self.model(img=imgs, classes=masks)
            
        self.accelerator.backward(loss)        
                        
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

        self.opt.step()
        self.opt.zero_grad()
        self.scheduler.step()

        # Log metrics to TensorBoard
        if self.accelerator.is_local_main_process:
            self.writer.add_scalar('Loss/train', loss.item(), self.step)
            self.writer.add_scalar('Learning_rate', self.scheduler.get_lr()[0], self.step)

        return loss
    
    def save_colored_masks(self, masks, save_path, nrow):  # Add self parameter
        """Convert masks to colored visualization and save as image."""
        # Define colors for different mask values (adjust colors as needed)
        colors = {
            0: '#FFFFFF',  # white
            1: '#FFA500',  # orange
            2: '#00FFFF',  # cyan
            3: '#FF0000',  # red
            4: '#008000'   # green
        }
        
        # Calculate grid dimensions
        n_samples = masks.shape[0]
        n_rows = int(math.sqrt(n_samples))
        n_cols = math.ceil(n_samples / n_rows)
        
        # Create figure with subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1 or n_cols == 1:
            axs = axs.reshape(n_rows, n_cols)
        
        # Plot each mask
        for idx in range(n_samples):
            row = idx // n_cols
            col = idx % n_cols
            
            # Create colormap from unique values in mask
            unique_values = torch.unique(masks[idx])
            cmap = ListedColormap([colors[val.item()] for val in unique_values])
            
            # Plot mask
            axs[row, col].imshow(masks[idx, 0].cpu().detach(), cmap=cmap)
            axs[row, col].axis('off')
        
        # Remove empty subplots
        for idx in range(n_samples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            try:
                fig.delaxes(axs[row, col])
            except:
                pass
        
        # Save plot to file
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def eval_loop(self):
        if self.accelerator.is_main_process:
            # Get unwrapped models
            
            unwrapped_ema = self.accelerator.unwrap_model(self.ema)
            
            unwrapped_ema.to(self.accelerator.device)
            unwrapped_ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                # No need to unwrap ema_model again since we already have it unwrapped
                unwrapped_ema.ema_model.eval()

                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                    test_images, test_masks = next(self.test_loader)
                    
                    # Denormalize the test images before saving
                    test_images_denorm = test_images * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
                    
                    # Unwrap VAE model as before
                    vae = self.accelerator.unwrap_model(self.vae)
                    z = vae.encode(test_images[:self.num_samples]).latent_dist.sample() * self.norm_scale # correct scale
                    z = unwrapped_ema.ema_model.sample(z, test_masks[:self.num_samples]) / self.norm_scale  # correct scale  
                    test_samples = torch.clip(vae.decode(z).sample, -1, 1)  # Clip to [-1,1] range
                    test_samples = test_samples * 0.5 + 0.5  # Denormalize to [0,1] range
                    
                    utils.save_image(test_images_denorm[:self.num_samples], 
                                 str(self.results_folder / f'images-{milestone}.png'), 
                                 nrow = int(math.sqrt(self.num_samples)))   
                    
                    self.save_colored_masks(test_masks[:self.num_samples], 
                               str(self.results_folder / f'masks-{milestone}.png'), 
                               nrow = int(math.sqrt(self.num_samples)))         
                    
                    utils.save_image(test_samples, 
                                 str(self.results_folder / f'sample-{milestone}.png'), 
                                 nrow = int(math.sqrt(self.num_samples)))
                    
                    if self.debug:
                        print("\n=== Generated Samples Details [Trainer.eval_loop()] ===")
                        print(f"Test images metadata:")
                        print(f"  Shape: {test_images.shape}")
                        print(f"  Dtype: {test_images.dtype}")
                        print(f"  Device: {test_images.device}")
                        print(f"  Value range: [{test_images.min():.2f}, {test_images.max():.2f}]")
                        print(f"\nTest masks metadata:")
                        print(f"  Shape: {test_masks.shape}")
                        print(f"  Dtype: {test_masks.dtype}")
                        print(f"  Device: {test_masks.device}")
                        print(f"  Unique values: {torch.unique(test_masks).tolist()}")
                        print(f"\nGenerated samples metadata:")
                        print(f"  Shape: {test_samples.shape}")
                        print(f"  Dtype: {test_samples.dtype}")
                        print(f"  Device: {test_samples.device}")
                        print(f"  Value range: [{test_samples.min():.2f}, {test_samples.max():.2f}]")
                    
                    # Add sample images to TensorBoard
                    self.writer.add_images('Generated_samples', test_samples, self.step)
                    self.writer.add_images('Input_images', test_images_denorm, self.step)
                    
                    # Convert masks to RGB for visualization
                    mask_rgb = torch.zeros((test_masks.size(0), 3, test_masks.size(2), test_masks.size(3)), 
                                        device=test_masks.device)
                    for i in range(self.model.num_classes):
                        mask_rgb[:, i % 3] += (test_masks == i).float()
                    self.writer.add_images('Masks', mask_rgb, self.step)
                    
                    # Only save model checkpoint at milestone intervals
                    if self.step % self.save_milestone_every == 0:
                        self.save(milestone)

    def train(self):

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data,masks=next(self.dl)
                    
                    with self.accelerator.accumulate(self.model):
                        loss = self.train_loop(data,masks)
                        total_loss += loss.item()

                total_loss/=self.gradient_accumulate_every
                if self.step % self.save_loss_every == 0:
                    self.running_loss.append(total_loss)
                    self.running_lr.append(self.scheduler.get_lr()[0])

                pbar.set_description(f'loss: {total_loss:.4f}')

                self.step += 1
                self.eval_loop()
                pbar.update(1)

        self.accelerator.print('training complete')

        # Close TensorBoard writer
        if self.accelerator.is_local_main_process:
            self.writer.close()