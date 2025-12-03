import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import logging
from datetime import datetime
import imageio.v2 as imageio
import Augmentor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from datasets.get_dataset import dataset
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm import tqdm
import src.metrics as metrics
from torch.optim.lr_scheduler import CosineAnnealingLR

 

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])


# helpers functions


def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

def hu_to_save_image(tensor_img):
    # 反归一化（[-1024, 3072]）
    img = metrics.denormalize_(tensor_img)
    # 截断到 [-160, 240]
    img = metrics.trunc(img)
    # 映射到 [0,1] 用于 save_image
    img = (img + 160.0) / 400.0
    return img.clamp(0, 1)


def save_a_img(img, path):  # function of save $img$ (np.ndarray of range [0.0, 1.0]) to $path$
    return imageio.imwrite(path, (img * 65535).astype(np.uint16))


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float64)
    else:
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float64)
    assert alphas.sum() - torch.tensor(1) < torch.tensor(1e-10)

    return alphas * sum_scale


class ResidualDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_res_noise',
            ddim_sampling_eta=0.,
            condition=False,
            sum_scale=None,
            input_condition=False,
            input_condition_mask=False
    ):
        super().__init__()
        assert not (
                type(self) == ResidualDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        alphas = gen_coefficients(timesteps, schedule="decreased")
        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2 = gen_coefficients(
            timesteps, schedule="increased", sum_scale=self.sum_scale)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
        betas_cumsum = torch.sqrt(betas2_cumsum)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val):
            return self.register_buffer(
                name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1 - alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev / betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                                                 alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2 / betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
                (x_t - x_input - (extract(self.alphas_cumsum, t, x_t.shape) - 1)
                 * pred_res) / extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
                (x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_input -
                 extract(self.betas_cumsum, t, x_t.shape) * noise) / extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
                x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_res -
                extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t - extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape) / extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None,
                          clip_denoised=True): 
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_res_add_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input - model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == 'pred_x0_add_noise':
            x_start = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res) * 0
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            x_start = model_output[0]
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res) * 0
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
           # img = x_input+math.sqrt(self.sum_scale) * \
           #    torch.randn(shape, device=device)
           # input_add_noise = img    #if you want try RDDM use this 
            img = x_input  #  our method
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [x_input] + img_list
            else:
                img_list = [x_input, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            # img = x_input+math.sqrt(self.sum_scale) * \
            #    torch.randn(shape, device=device)
            # input_add_noise = img    #if you want try RDDM use this 
            img = x_input  #  our method
        else:
            img = torch.randn(shape, device=device)

        x_start = None  
        type = "use_pred_noise"

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, img, time_cond, x_input_condition, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                                                                                betas2_cumsum_next - sigma2).sqrt() / betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if type == "use_pred_noise":
                # img = img - alpha*pred_res - \
                #    (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
                #    pred_noise + sigma2.sqrt()*noise    #if you want try RDDM use this
                img = img - alpha * pred_res  #  our method

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [x_input] + img_list
            else:
                img_list = [x_input, img]  
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res +
                extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  # Condition
            if self.input_condition:
                x_input_condition = imgs[2]
            else:
                x_input_condition = 0
            x_input = imgs[1]
            x_start = imgs[0]  # gt = imgs[0], input = imgs[1]
        else:  # Generation
            x_input = 0
            x_start = imgs

        noise = default(noise, lambda: torch.randn_like(x_start)) * 0  # noise = default(noise, lambda: torch.randn_like(x_start)) RDDM use this
        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:  # and random.random() < 0.5
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_input, x, t, x_input_condition if self.input_condition else 0).pred_x_start
                x_self_cond.detach_()
        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in,
                               t,
                               x_self_cond)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_res_add_noise':
            target.append(x_res)
            target.append(x_res + noise)

            pred_res = model_out[0]
            pred_noise = model_out[1] - model_out[0]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input - model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_add_noise':
            target.append(x_start)
            target.append(x_start + noise)

            pred_res = x_input - model_out[0]
            pred_noise = model_out[1] - model_out[0]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)

            pred_res = model_out[0]
        elif self.objective == "pred_x0":
            target.append(x_start)
            pred_x0 = model_out[0]
            pred_res = x_input - model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000 * self.loss_fn(x_u, u_gt, reduction='none')
        else:
            loss = 0
            for i in range(len(model_out)):
                loss = loss + \
                       self.loss_fn(model_out[i], target[i], reduction='none') 
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if self.input_condition and self.input_condition_mask:
            img[0] = normalize_to_neg_one_to_one(img[0])
            img[1] = normalize_to_neg_one_to_one(img[1])
        else:
            img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)


# trainer class


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            augment_flip=True,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='/root/autodl-tmp/ResGMDiff-main/result',
            amp=False,
            fp16=False,
            split_batches=True,
            convert_image_to=None,
            condition=False,
            sub_dir=False,
            equalizeHist=False,
            crop_patch=False,
            generation=False
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition

        log_dir = '/root/autodl-tmp/ResGMDiff-main/logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger()

        if self.condition:
            if len(folder) == 3:
                self.condition_type = 1
                # test_input
                ds = dataset(folder[-1], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=0,
                             equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.sample_loader = cycle(
                    self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                        pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist,
                             crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                                                    shuffle=True, pin_memory=True, num_workers=4)))
            elif len(folder) == 4:
                self.condition_type = 2
                # test_gt+test_input
                ds = dataset(folder[2:4], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=1,
                             equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.sample_loader = cycle(
                    self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                        pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist,
                             crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                                                    shuffle=True, pin_memory=True, num_workers=4, drop_last=True)))
            elif len(folder) == 6:
                self.condition_type = 3
                # test_gt+test_input
                ds = dataset(folder[3:6], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=2,
                             equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:3]

                self.sample_dataset = ds
                self.sample_loader = cycle(
                    self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                        pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist,
                             crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                                                    shuffle=True, pin_memory=True, num_workers=4)))
        else:
            self.condition_type = 0
            trian_folder = folder

            ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                         convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist,
                         crop_patch=crop_patch, generation=generation)
            self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                                                shuffle=True, pin_memory=True, num_workers=4)))

        self.logger.info('Initial Dataset Finished')

        # optimizer

        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr, betas=adam_betas)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #                 self.opt,
        #                 T_max=1200,       
        #                 eta_min=2e-6         
        #                 )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0
        self.epoch = 0
        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        device = self.accelerator.device
        self.device = device
        self.logger.info('Initial Model Finished')

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'best_model.pt'))

    def load(self, milestone):
        path = Path('/root/autodl-tmp/ResGMDiff-main/result/best_model.pt')

        if path.exists():
            data = torch.load(
                str(path), map_location=self.device)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - " + str(path))

        self.ema.to(self.device)

    def train(self):
        accelerator = self.accelerator
        self.best_psnr = 0.0
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss = total_loss + loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                # if self.step%100 == 0:
                #     self.scheduler.step()
                if self.step * self.batch_size % 5410 == 0:
                    self.epoch += 1

                
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()
                    # val
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        result_path = '/root/autodl-tmp/ResGMDiff-main/result/train_val/{}'.format(milestone)
                        os.makedirs(result_path, exist_ok=True)
                        org_avgpsnr, pred_avgpsnr, org_avgssim, pred_avgssim, org_avgrmse, pred_avgrmse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        for i in range(10):
                            original_psnr, original_ssim, original_rmse, pred_psnr, pred_ssim, pred_rmse = self.sample(
                                milestone, result_path, i)
                            org_avgpsnr = org_avgpsnr + original_psnr
                            pred_avgpsnr = pred_avgpsnr + pred_psnr

                            org_avgssim = org_avgssim + original_ssim
                            pred_avgssim = pred_avgssim + pred_ssim

                            org_avgrmse = org_avgrmse + original_rmse
                            pred_avgrmse = pred_avgrmse + pred_rmse
                        print('avgpsr %.4f to %.4f; avgssim %.3f to %.3f; avgrmes %.3f to %.3f' % (
                            org_avgpsnr / 10.0, pred_avgpsnr / 10.0, org_avgssim / 10.0,
                            pred_avgssim / 10.0, org_avgrmse / 10.0, pred_avgrmse / 10.0))
                        self.logger.info(
                            f'epoch: {self.epoch} step: {self.step} milestone: {milestone} avgpsnr{org_avgpsnr / 10.0:.4f} to {pred_avgpsnr / 10.0:.4f}')

                        if self.step != 0 and self.step % (self.save_and_sample_every * 2) == 0: #这里要先改成1原来是5
                            if self.best_psnr < pred_avgpsnr:
                                self.best_psnr = pred_avgpsnr
                                self.save(milestone)
                                self.logger.info(
                                    f'save the best model milestone: {milestone} epoch: {self.epoch} step: {self.step} ')
                if self.step % 200 == 0:
                    self.logger.info(f'epoch: {self.epoch} step: {self.step} loss: {total_loss:.4f}')
                pbar.set_description(f'loss: {total_loss:.4f}')
                pbar.update(1)

        accelerator.print('training complete')

    def sample(self, milestone, result_path, j, last=True, FID=False):
        self.ema.ema_model.eval()

        with torch.no_grad():
            batches = self.num_samples
            if self.condition_type == 0:
                x_input_sample = [0]
                show_x_input_sample = []
            elif self.condition_type == 1:
                x_input_sample = [next(self.sample_loader).to(self.device)]
                show_x_input_sample = x_input_sample
            elif self.condition_type == 2:  # 我们使用2
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                x_taeget_sample = x_input_sample[0]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]
            # elif self.condition_type == 3:
            #    x_input_sample = next(self.sample_loader)
            #    x_input_sample = [item.to(self.device)
            #                      for item in x_input_sample]
            #    show_x_input_sample = x_input_sample
            #    x_input_sample = x_input_sample[1:]

            all_output_sample = self.ema.ema_model.sample(
                x_input_sample, batch_size=batches, last=last)
            output_sample = all_output_sample[1]
            self.org_avgpsnr, self.pred_avgpsnr, self.org_avgssim, self.pred_avgssim, self.org_avgrmse, self.pred_avgrmse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for i in range(batches):
                input_tensor = x_input_sample[i].squeeze(0).cpu()
                target_tensor = x_taeget_sample.squeeze(0).cpu()
                output_tensor = output_sample[i].squeeze(0).cpu()
                output_npy = output_sample[i].squeeze(0).cpu().numpy()
                input_tensor = metrics.trunc(metrics.denormalize_(input_tensor))
                target_tensor = metrics.trunc(metrics.denormalize_(target_tensor))
                output_tensor = metrics.trunc(metrics.denormalize_(output_tensor))
                (original_psnr, original_ssim, original_rmse), (
                    pred_psnr, pred_ssim, pred_rmse) = metrics.compute_measure(input_tensor, target_tensor,
                                                                               output_tensor, 400)

                # save_a_img(output_npy, os.path.join(result_path, str(j) + '.png') )
                print('%d-, psnr %.4f to %.4f; ssim %.3f to %.3f, rmes %.2f to %.2f' % (
                    i, original_psnr, pred_psnr, original_ssim, pred_ssim, original_rmse, pred_rmse))

            all_images_list = show_x_input_sample + \
                              list(all_output_sample)
            all_images_list = [hu_to_save_image(t.squeeze(0).cpu()) for t in all_images_list]
            all_images = torch.stack(all_images_list, dim=0)
            if last:
                nrow = int(math.sqrt(self.num_samples))
            else:
                nrow = all_images.shape[0]

            if FID:
                for i in range(batches):
                    file_name = f'sample-{milestone}.png'
                    utils.save_image(
                        all_images_list[0][i].unsqueeze(0), os.path.join(result_path, file_name), nrow=1)
                    milestone += 1
                    if milestone >= self.total_n_samples:
                        break
            else:
                file_name = f'sample-{milestone}-{j}.png'
                utils.save_image(all_images, os.path.join(result_path, file_name), nrow=nrow)
            print("sampe-save " + file_name)
            print("sampe-save " + file_name)
        # return milestone
        return original_psnr, original_ssim, original_rmse, pred_psnr, pred_ssim, pred_rmse

    @torch.no_grad()
    def test(self, milestone, sample=False, last=True):
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
        result_path = '/root/autodl-tmp/ResGMDiff-main/result/test_val'
        os.makedirs(result_path, exist_ok=True)
        org_psnr_list, pred_psnr_list = [], []
        org_ssim_list, pred_ssim_list = [], []
        org_rmse_list, pred_rmse_list = [], []
        print('lens:', len(self.sample_dataset))
        for i in range(len(self.sample_dataset)):
            original_psnr, original_ssim, original_rmse, pred_psnr, pred_ssim, pred_rmse = self.sample(milestone,
                                                                                                       result_path, i)
            org_psnr_list.append(original_psnr)
            pred_psnr_list.append(pred_psnr)

            org_ssim_list.append(original_ssim)
            pred_ssim_list.append(pred_ssim)

            org_rmse_list.append(original_rmse)
            pred_rmse_list.append(pred_rmse)

        org_psnr_list, pred_psnr_list = np.array(org_psnr_list), np.array(pred_psnr_list)
        org_ssim_list, pred_ssim_list = np.array(org_ssim_list), np.array(pred_ssim_list)
        org_rmse_list, pred_rmse_list = np.array(org_rmse_list), np.array(pred_rmse_list)
        print('avgpsnr %.4f to %.4f±%.4f; avgssim %.4f to %.4f±%.4f; avgrmse %.4f to %.4f±%.4f' % (
        org_psnr_list.mean(), pred_psnr_list.mean(), pred_psnr_list.mean() - pred_psnr_list.min(),
        org_ssim_list.mean(), pred_ssim_list.mean(), pred_ssim_list.mean() - pred_ssim_list.min(),
        org_rmse_list.mean(), pred_rmse_list.mean(), pred_rmse_list.mean() - pred_rmse_list.min()
        ))
        print('test end')
        
    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)
