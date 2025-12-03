import os
import sys
import torch
from src.MR-Net import (Unet, UnetRes)
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, set_seed)
from ptflops import get_model_complexity_info
# PYTHONPATH=$(pwd) python ResGMDiff-main/train.py 

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
sys.stdout.flush()
set_seed(10)
debug = False
if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 1000#1000
    if len(sys.argv)>1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 5 
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 100000



condition = True
input_condition = False#False  
input_condition_mask = False

if condition:
    if input_condition:
        folder = ["/root/autodl-tmp/Data/CT_train",
                "/root/autodl-tmp/Data/CT_train",
	"/root/autodl-tmp/Data/CT_cond",  
                "/root/autodl-tmp/Data/CT_cond",
                "/root/autodl-tmp/Data/CT_val",   
                "/root/autodl-tmp/Data/CT_val"]
    else:
        folder = ["/root/autodl-tmp/Data/npy_img_1mm_train",
                "/root/autodl-tmp/Data/npy_img_1mm_train",
                "/root/autodl-tmp/Data/npy_img_1mm_val",   
                "/root/autodl-tmp/Data/npy_img_1mm_val"]   
    train_batch_size = 1
    num_samples = 1
    sum_scale = 0.01  
    image_size = 512
else:
    folder = 'generation_CT_datasets'
    train_batch_size = 4
    num_samples = 25
    sum_scale = 1
    image_size = 512

model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        share_encoder=-1, 
        condition=condition,
        input_condition=input_condition
)
#model = Network(in_channels=2)
diffusion = ResidualDiffusion(
     model,
     image_size=image_size,
     timesteps=5,           # number of steps 
     # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
     sampling_timesteps=sampling_timesteps,
     objective='pred_x0',
     loss_type='l1',            # L1 or L2
     condition=condition,
     sum_scale = sum_scale,
     input_condition=input_condition,
     input_condition_mask=input_condition_mask
)

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr= 8e-5,#8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,     # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to= None,
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation = False
)




#if not trainer.accelerator.is_local_main_process:
#    pass
#else:
#trainer.load(60)

# train
trainer.train()




