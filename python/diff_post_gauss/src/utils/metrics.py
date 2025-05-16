from typing import Tuple
from pathlib import Path

import torch
from torch import Tensor
import numpy as np
import lpips
from utils.utils import normalize_tensor
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance


class SSIM:
    def __init__(self, window_size=11, size_average=False) -> None:
        # keep same as Lama
        self.ssim = SSIMScore(window_size=window_size, size_average=False).eval()

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # samples: B, C, H, W
        # references: 1, C, H, W or B, C, H, W
        B = samples.shape[0]
        samples = normalize_tensor(samples)
        references = normalize_tensor(references)
        if references.shape[0] == 1:
            references = references.repeat(B, 1, 1, 1)
        return self.ssim(samples, references).detach().cpu()


# This takes (0., 1.) images
class SSIMScore(torch.nn.Module):
    """SSIM. Modified from:
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer("window", self._create_window(window_size, self.channel))

    def forward(self, img1, img2):
        assert len(img1.shape) == 4

        channel = img1.size()[1]

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            # window = window.to(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                np.exp(-((x - (window_size // 2)) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=(window_size // 2), groups=channel)
        mu2 = F.conv2d(img2, window, padding=(window_size // 2), groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=(window_size // 2), groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=(window_size // 2), groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=(window_size // 2), groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()

        return ssim_map.mean(1).mean(1).mean(1)


class PSNR:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # samples: B, C, H, W
        # references: 1, C, H, W or B, C, H, W
        B = samples.shape[0]
        samples = normalize_tensor(samples)
        references = normalize_tensor(references)
        if references.shape[0] == 1:
            references = references.repeat(B, 1, 1, 1)

        mse = torch.mean((samples - references) ** 2, dim=(1, 2, 3))
        peak = 1.0  # we normalize the image to (0., 1.)
        psnr = 10 * torch.log10(peak / mse)
        return psnr.detach().cpu()


def check_device(tensor, device):
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def check_image(tensor):
    assert torch.max(tensor) <= 1.0 + 1e-3 and torch.min(tensor) >= -1.0 - 1e-3


class LPIPS:
    def __init__(self, base_model="alex", device="cpu") -> None:
        self.device = device
        self.loss_fn = lpips.LPIPS(net=base_model).to(device)

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # ! Notice that samples and references should be in [-1, 1]
        check_image(samples)
        check_image(references)
        samples = check_device(samples, self.device)
        references = check_device(references, self.device)
        return self.loss_fn(samples, references).detach().cpu()

    def on_dir():
        pass


class FID:
    def __init__(
        self,
        path_real_imgs: str | Path,
        batch_size: int = 50,
        dims: int = 2048,
        num_workers: int = 1,
        device: str = "cpu",
    ) -> None:
        self.path_real_imgs = path_real_imgs
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device

        # load Inception V3 model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)

        # compute stats of real data
        self.mean_real, self.cov_real = self._compute_stats(path_real_imgs)

    def compute_FID(self, path_imgs: str | Path) -> float:
        # compute stat of data
        mean, cov = self._compute_stats(path_imgs)

        return calculate_frechet_distance(self.mean_real, self.cov_real, mean, cov)

    def _compute_stats(self, path_imgs: str) -> Tuple[Tensor, Tensor]:
        # NOTE cast path as str as `pytorch_fid` assumes it so
        mean, cov = compute_statistics_of_path(
            str(path_imgs),
            self.model,
            self.batch_size,
            self.dims,
            self.device,
            self.num_workers,
        )
        return mean, cov
