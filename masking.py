import math
from torchvision import transforms
import torch as th

from multi_resizer import MultiResizer
from resizer import Resizer
from utils import FilteredIfft2, Fft2, BlurOutwards, Identity

BLUR_KERNEL_SIGMA = 3.72  # 0.1% max height of gaussian -> sqrt(-2*log(0.1%))  # TODO: is this good?

def get_blend_operator_and_mask(mask, blend_pix):
    if (blend_pix is None) or (mask is None):
        blend_mask = 1  # use only non-masked - normal ILVR
        blend = Identity()
    else:
        blend = BlurOutwards(n_pixels=blend_pix) if blend_pix > 0 else Identity()
        blend_mask = 1 - blend(1 - mask)
    return blend_mask, blend

def get_low_pass_operator(down_N, blur_sigma, shape, device):
    assert ((blur_sigma is None) or (down_N is None)), 'Must choose a single filtering method (down_N/blur_sigma)'
    if down_N is not None:
        shape_d = list(shape)
        shape_d[2] = int(shape_d[2] / down_N)
        shape_d[3] = int(shape_d[3] / down_N)
        down = Resizer(shape, 1 / down_N).to(device)
        up = Resizer(tuple(shape_d), down_N).to(device)
    elif blur_sigma is not None and blur_sigma > 0:
        kernel_size = 2 * math.ceil(BLUR_KERNEL_SIGMA * blur_sigma) + 1
        down = transforms.GaussianBlur(kernel_size, sigma=blur_sigma)
        up = Identity()
    else:
        down = Identity()
        up = Identity()
    return down, up

def shift_mask(mask, a, b, device):
    if (mask is None) or (a is None) or (b is None):
        mask = th.ones((1, 1, 1, 1)).to(device)
    else:
        mask = (b - a) * mask + a  # map values from [0, 1] to [a, b]
    return mask

def get_fft_operator_and_mask(mask, fft_num, shape, device):
    if (fft_num is None) or (mask is None):
        freq_mask = th.ones((1, 1, 1, 1)).to(device)
        up, down = None, None
    else:
        n_freq_bands = 3
        down = Fft2(shape, True).to(device)
        up = FilteredIfft2(shape, n_freq_bands, True).to(device)
        active_freq = format(fft_num, f'#0{n_freq_bands + 2}b')[2:]
        # TODO: find better solution to create mask
        freq_mask_tmp = th.floor(th.abs((mask - 1e-7)) * n_freq_bands)  # discretize from [0, 1]
        freq_mask = th.zeros((shape[0], n_freq_bands, shape[2], shape[3])).to(device)
        freq_mask[:, [0], :, :] = th.where(freq_mask_tmp <= 0.2, float(active_freq[-1]), 1.)
        freq_mask[:, [1], :, :] = th.where(freq_mask_tmp <= 0.2, float(active_freq[-2]), 1.)
        freq_mask[:, [2], :, :] = th.where(freq_mask_tmp <= 0.2, float(active_freq[-3]), 1.)
    return down, up, freq_mask

# TODO: temp
def create_masks(mask,
                 blend_pix,
                 down_N_out,
                 down_N_in,
                 blur_sigma_out,
                 blur_sigma_in,
                 T_mask,
                 num_timesteps,
                 fft_num,
                 shape,
                 device):
    # blend operator for mask smoothing
    blend_mask, blend = get_blend_operator_and_mask(mask, blend_pix)

    # non-mask low-pass filter
    down_outer, up_outer = get_low_pass_operator(down_N_out, blur_sigma_out, shape, device)

    # mask low-pass filter
    down_inner, up_inner = get_low_pass_operator(down_N_in, blur_sigma_in, shape, device)

    # time mask (when to stop ILVR for each pixel)
    time_stop_mask = shift_mask(mask, (1 - T_mask) * num_timesteps, 0, device)

    # alpha mask (magnitude of ILVR per pixel)  # TODO: maybe we can deprecate this
    # mask = shift_mask(mask, mask_alpha, 1, device)

    # TODO: maybe we can deprecate this
    down_fft, up_fft, freq_mask = get_fft_operator_and_mask(mask, fft_num, shape, device)
    if (down_fft is not None) and (up_fft is not None):
        down_inner = down_fft
        up_inner = up_fft

    # # TODO: maybe we can deprecate this
    # if (down_N_in is None) or (mask is None):
    #     shape_d = list(shape)
    #     shape_d[2] = int(shape_d[2] / down_N_in)
    #     shape_d[3] = int(shape_d[3] / down_N_in)
    #     down_inner = Resizer(shape, 1 / down_N_in).to(device)
    #     up_inner = Resizer(tuple(shape_d), down_N_in).to(device)
    # else:
    #     N_mask = (down_N_out - down_N_out) * mask + down_N_in  # convert from [0, 1] to [down_N_out, down_N_in]
    #     N_mask = (2 ** th.round(th.log2(down_N_in))).int()
    #     down_inner = Identity().to(device)
    #     up_inner = MultiResizer(N_mask, shape, device).to(device)

    return
