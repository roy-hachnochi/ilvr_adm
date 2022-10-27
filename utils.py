import torch
from torch import nn
from torch.nn import functional as f

from resizer import Resizer

# ======================================================================================================================
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        return in_tensor

# ======================================================================================================================
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

# ======================================================================================================================
FFT_RESIZING = 2

class FilteredIfft2(nn.Module):
    def __init__(self, shape, n_freq_bands, downsample=False):
        super(FilteredIfft2, self).__init__()
        if downsample and (FFT_RESIZING > 1):
            shape_d = list(shape)
            shape_d[2] = int(shape_d[2] * FFT_RESIZING)
            shape_d[3] = int(shape_d[3] * FFT_RESIZING)
            self.resizer = Resizer(tuple(shape_d), 1 / FFT_RESIZING)
        else:
            self.resizer = Identity()
        self.n_freq_bands = n_freq_bands
        self.shape = shape

    def forward(self, in_tensor, f_mask, *ignore_args, **ignore_kwargs):
        h, w = in_tensor.shape[-2:]
        in_tensor = torch.fft.fftshift(in_tensor)
        out_tensor = torch.zeros(self.shape).to(in_tensor.device)
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        x, y = x.to(out_tensor.device), y.to(out_tensor.device)
        for b in range(self.n_freq_bands):
            if b == 0:
                clip_vals = (0, 0.4 * (0.5 ** (self.n_freq_bands - b - 1)))
            else:
                clip_vals = (0.4 * (0.5 ** (self.n_freq_bands - b)), 0.4 * (0.5 ** (self.n_freq_bands - b - 1)))
            mask = ((y >= -clip_vals[1]) & (y <= clip_vals[1]) & (x >= -clip_vals[1]) & (x <= clip_vals[1])) & ~(
                    (y > -clip_vals[0]) & (y < clip_vals[0]) & (x > -clip_vals[0]) & (x < clip_vals[0]))
            reconstruction = self.resizer(torch.fft.ifft2(torch.fft.ifftshift(mask * in_tensor)))
            out_tensor = out_tensor + f_mask[:, [b]] * torch.real(reconstruction)
        return out_tensor


class Fft2(nn.Module):
    def __init__(self, shape, upsample=False):
        super(Fft2, self).__init__()
        if upsample and (FFT_RESIZING > 1):
            self.resizer = Resizer(shape, FFT_RESIZING)
        else:
            self.resizer = Identity()

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        return torch.fft.fft2(self.resizer(in_tensor))
