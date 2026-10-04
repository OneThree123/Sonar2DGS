#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def weighted_mean(value, weight, eps=1e-8):
    weight = torch.clamp(weight.to(device=value.device, dtype=value.dtype), min=0.0)
    return (value * weight).sum() / torch.clamp(weight.sum(), min=eps)

def masked_l1_loss(network_output, gt, mask, eps=1e-8):
    return weighted_mean(torch.abs(network_output - gt), mask, eps=eps)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def gaussian_blur(img, window_size=3, sigma=1.0):
    if img.ndim == 2:
        batched = img[None, None]
        restore = lambda x: x[0, 0]
    elif img.ndim == 3:
        batched = img[None]
        restore = lambda x: x[0]
    elif img.ndim == 4:
        batched = img
        restore = lambda x: x
    else:
        raise ValueError(f"Unsupported image shape for gaussian blur: {tuple(img.shape)}")

    channel = batched.size(-3)
    kernel_1d = gaussian(window_size, sigma).to(device=img.device, dtype=img.dtype).unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).float().unsqueeze(0).unsqueeze(0)
    kernel = kernel_2d.to(device=img.device, dtype=img.dtype)
    kernel = kernel.expand(channel, 1, window_size, window_size).contiguous()
    blurred = F.conv2d(batched, kernel, padding=window_size // 2, groups=channel)
    return restore(blurred)

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def masked_ssim(img1, img2, mask, window_size=11, size_average=True, eps=1e-8):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mask = torch.clamp(mask.to(device=img1.device, dtype=img1.dtype), 0.0, 1.0)
    if mask.size(-3) == 1 and channel != 1:
        mask = mask.expand(*mask.shape[:-3], channel, mask.shape[-2], mask.shape[-1])

    ssim_map = _masked_ssim_map(img1, img2, mask, window, window_size, channel, eps)
    if size_average:
        return ((ssim_map * mask).sum() + eps) / (mask.sum() + eps)
    return ssim_map.mean(1).mean(1).mean(1)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _masked_ssim_map(img1, img2, mask, window, window_size, channel, eps=1e-8):
    local_weight = F.conv2d(mask, window, padding=window_size // 2, groups=channel)
    local_weight = torch.clamp(local_weight, min=eps)

    mu1 = F.conv2d(img1 * mask, window, padding=window_size // 2, groups=channel) / local_weight
    mu2 = F.conv2d(img2 * mask, window, padding=window_size // 2, groups=channel) / local_weight

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1 * mask, window, padding=window_size // 2, groups=channel) / local_weight - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2 * mask, window, padding=window_size // 2, groups=channel) / local_weight - mu2_sq
    sigma12 = F.conv2d(img1 * img2 * mask, window, padding=window_size // 2, groups=channel) / local_weight - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
