import torch
from typing import Tuple, Callable, Union
import tqdm
import numpy as np


class GenericSampler:

    def __init__(self, tqdm_disable=True, **kwargs):

        self.tqdm_disable = tqdm_disable

    def create_step_fn(
        self,
        denoiser_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ],
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:

        "To be implemented"
        raise NotImplementedError("Create step fn has to be implemented")
        return None

    def sample(
        self,
        stds: torch.Tensor,
        initial_samples: torch.Tensor,
        denoiser_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ]
    ) -> torch.Tensor:
        step_fun = self.create_step_fn(denoiser_fn)

        sample = initial_samples
        stds_flipped = torch.flip(stds, dims=(0,)).to(dtype=torch.float64).to(initial_samples.device)
        pbar = tqdm.tqdm(
            zip(stds_flipped[:-1], stds_flipped[1:]), disable=self.tqdm_disable
        )
        for sigma_t, sigma_t_prev in pbar:
            sample = step_fun(sample, sigma_t, sigma_t_prev)
        return denoiser_fn(sample, stds[0])


class DDIM(GenericSampler):
    """
    The forward process is: X_t = X_s + \sigma_{t|s} Z
    """

    def __init__(self, eta=0.0, **kwargs):
        """
        assumes that the model has been trained using variance exploding
        """
        super().__init__(**kwargs)
        self.eta = eta
        

    def create_step_fn(
        self,
        denoiser_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ],
        **kwargs
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        def _step(
            x: torch.Tensor, sigma_t: torch.Tensor, sigma_t_prev: torch.Tensor, **kwargs
        ) -> torch.Tensor:
            std = (sigma_t_prev / sigma_t) * ((sigma_t**2 - sigma_t_prev**2)**.5) * self.eta
            pred_x0 = denoiser_fn(x, sigma_t)
            coeff_residue = ((sigma_t_prev**2  - std**2)**.5) / sigma_t
            return pred_x0 + coeff_residue * (x - pred_x0) + std * torch.randn_like(x)

        return _step


def create_karras_sigmas(
    N: int, rho: float = 7.0, sigma_min: float = 0.002, sigma_max: float = 80.0
) -> torch.Tensor:
    stds = (
        sigma_max ** (1 / rho)
        + torch.linspace(1, 0, N) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return stds


class EDM(GenericSampler):
    def __init__(
        self,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        dtype=torch.float32,
        randn_like=torch.randn_like,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.randn_like = randn_like

    def create_step_fn(
        self,
        denoiser_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ]
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        def _step(
            x: torch.Tensor,
            sigma_t: torch.Tensor,
            sigma_t_prev: torch.Tensor,
        ) -> torch.Tensor:
            x_cur = x

            # # Increase noise temporarily.
            # if self.S_churn > 0 and self.S_min <= sigma_t <= self.S_max:
            #     gamma = min(self.S_churn / num_steps, np.sqrt(2) - 1)
            #     t_hat = sigma_t + gamma * sigma_t
            #     x_hat = x_cur + (t_hat ** 2 - sigma_t ** 2).sqrt() * self.S_noise * randn_like(x_cur)
            # else:
            #     t_hat = sigma_t
            #     x_hat = x_cur
            t_hat = sigma_t
            x_hat = x_cur

            # Euler step.
            d_cur = (x_hat - denoiser_fn(x_hat, t_hat)) / t_hat
            x_next = x_hat + (sigma_t_prev - t_hat) * d_cur

            # # Apply 2nd order correction.
            # if i < num_steps - 1:
            d_prime = (
                x_next - denoiser_fn(x_next, sigma_t_prev)
            ) / sigma_t_prev
            x_next = x_hat + (sigma_t_prev - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            return x_next

        return _step
