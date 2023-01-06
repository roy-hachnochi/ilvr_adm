import math
from torchvision import transforms
import torch
from torch import nn
from torch.nn import functional as f
from kornia.color import rgb_to_lab, lab_to_rgb

from ResizeRight.resize_right import resize

BLUR_KERNEL_SIGMA = 3.72  # 0.1% max height of gaussian -> sqrt(-2*log(0.1%))  # TODO: is this good?

class Resizer(nn.Module):
    def __init__(self, down_N):
        super(Resizer, self).__init__()
        self.down_N = down_N

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        out_tensor = resize(in_tensor, scale_factors=1 / self.down_N)
        out_tensor = resize(out_tensor, scale_factors=self.down_N)
        return out_tensor

class Harmonization(nn.Module):
    def __init__(self, w):
        super(Harmonization, self).__init__()
        self.w = w

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        out_tensor = rgb_to_lab((in_tensor + 1.) / 2)  # rgb_to_lab expects [0, 1]
        out_tensor[:, 0, :, :] *= (1 - self.w)
        out_tensor = lab_to_rgb(out_tensor, clip=False) * 2 - 1  # back to [-1, 1]
        return out_tensor

class Colorization(nn.Module):
    def __init__(self, w):
        super(Colorization, self).__init__()
        self.w = w

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        out_tensor = rgb_to_lab((in_tensor + 1.) / 2)  # rgb_to_lab expects [0, 1]
        out_tensor[:, 1:, :, :] *= (1 - self.w)
        out_tensor = lab_to_rgb(out_tensor, clip=False) * 2 - 1  # back to [-1, 1]
        return out_tensor

def get_blend_operator_and_mask(mask, blend_pix):
    if (blend_pix is None) or (mask is None):
        blend_mask = 1  # use only non-masked - normal ILVR
        blend = Identity()
    else:
        blend = BlurOutwards(n_pixels=blend_pix) if blend_pix > 0 else Identity()
        blend_mask = blend(mask)
        # blend_mask = 1 - blend(1 - mask)
    return blend_mask, blend

def get_low_pass_operator(down_N, blur_sigma=None):
    assert ((blur_sigma is None) or (down_N is None)), 'Must choose a single filtering method (down_N/blur_sigma)'
    if down_N is not None:
        return Resizer(down_N)
    elif blur_sigma is not None and blur_sigma > 0:  # TODO: can we depreciate this?
        kernel_size = 2 * math.ceil(BLUR_KERNEL_SIGMA * blur_sigma) + 1
        return transforms.GaussianBlur(kernel_size, sigma=blur_sigma)
    else:
        return Identity()

def get_ilvr_operator(down_N, w, harmonization=False, colorization=False):
    phi = torch.nn.Sequential(get_low_pass_operator(down_N))
    if harmonization:
        phi.append(Harmonization(w))
    if colorization:
        phi.append(Colorization(w))
    return phi

def shift_mask(mask, a, b, T, device):
    if mask is None:
        mask = torch.ones((1, 1, 1, 1)).to(device)
    if a is None:
        a = 0
    if b is None:
        b = T
    mask = (b - a) * mask + a  # map values from [0, 1] to [a, b]
    return mask

# ======================================================================================================================
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        return in_tensor

class BlurOutwards(nn.Module):
    def __init__(self, n_pixels, smoother='linear'):
        super(BlurOutwards, self).__init__()
        self.n_pixels = n_pixels
        if smoother == 'sigmoid':
            self.smoother = torch.sigmoid(torch.linspace(5, -5, self.n_pixels + 1))
            self.smoother[0] = 1
            self.smoother[-1] = 0
        elif smoother == 'linear':
            self.smoother = torch.linspace(1, 0, self.n_pixels + 1)
        else:
            raise NotImplementedError(f'\'{smoother}\' smoothing method is not implemented.')
        self.smoother_diff = -torch.diff(self.smoother)

    def forward(self, in_tensor):
        strel = torch.ones((3, 3)).to(in_tensor.device)
        out_tensor = torch.zeros_like(in_tensor)
        for i in range(self.n_pixels):
            out_tensor = out_tensor + self.smoother_diff[i] * in_tensor
            in_tensor = self._dilation(in_tensor, strel, origin=(1, 1))
        return out_tensor

    def _dilation(self, image, strel, origin=(0, 0)):
        """
        Credit: https://stackoverflow.com/a/66496234/12384018
        """
        # first pad the image to have correct unfolding; here is where the origins is used
        image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1])
        # Unfold the image to be able to perform operation on neighborhoods
        image_unfold = f.unfold(image_pad, kernel_size=strel.shape)
        # Flatten the structural element since its two dimensions have been flatten when unfolding
        strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
        # Perform the greyscale operation; sum would be replaced by rest if you want erosion
        sums = image_unfold * strel_flatten  # + in source, more convenient to use * for max operation
        # Take maximum over the neighborhood
        result, _ = sums.max(dim=1)
        # Reshape the image to recover initial shape
        return torch.reshape(result, image.shape)
