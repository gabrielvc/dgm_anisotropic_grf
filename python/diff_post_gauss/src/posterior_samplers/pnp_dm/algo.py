from tqdm import trange

import torch
from torch import Tensor
from posterior_samplers.pnp_dm.utils import get_denoiser
from ddrm.functions.svd_replacement import H_functions

from posterior_samplers.diffusion_utils import EpsilonNet, LDM
from utils.im_invp_utils import InverseProblem
from utils.utils import display


def pnp_dm(
    initial_noise: Tensor,
    epsilon_net: EpsilonNet,
    inverse_problem: InverseProblem,
    rho_start: float,
    rho_decay_rate: float,
    rho_min: float,
    mode: str,
    use_ode: bool,
    force_use_langevin: bool = False,
    n_langevin_steps: int = 100,
    langevin_stepsize: float = 1e-4,
):
    """PNP-DM, Split-Gibbs based Sampler algorithm as described in [1].

    Implementation adapted from https://github.com/zihuiwu/PnP-DM-public/

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    inverse_problem : Tuple
        observation, degradation operator, and standard deviation of noise.

    epsilon_net: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    rho_start : float
        The initial intensity of the coupling parameter.

    rho_decay_rate : float
        The exponential rate of decay of the coupling parameter.

    rho_min : float
        The smallest intensity the coupling parameter can reach.

    mode : str
        Specifies how the denoiser is applied.
        Can be either "vp", "ve", "iddpm", or "edm".

    use_ode : bool
        When to denoise in a deterministic or stochastic way.

    n_langevin_steps : int, default=100
        The number of Langevin steps to perform when performing
        Langevin MC to peform the likelihood step.

    langevin_stepsize : float, default=0.0001
        The amplitude of Langevin stepsize.

    References
    ----------
    .. [1] Wu, Zihui, Yu Sun, Yifan Chen, Bingliang Zhang, Yisong Yue, and Katherine L. Bouman.
    "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors."(2024).
    """
    obs, H_func, std_obs = (
        inverse_problem.obs,
        inverse_problem.H_func,
        inverse_problem.std,
    )

    # LDM: Latent Diffusion Model
    is_LDM = isinstance(epsilon_net.net, LDM)

    # get denoiser
    denoiser = get_denoiser(epsilon_net, mode, use_ode)

    x = initial_noise
    n_steps = len(epsilon_net.timesteps) - 1
    for i in trange(n_steps):
        # annealing of the coupling parameter
        rho_iter = rho_start * (rho_decay_rate**i)
        rho_iter = max(rho_iter, rho_min)

        # likelihood step
        if (not force_use_langevin) and (not is_LDM) and _is_svd_decomposed(H_func):
            z = _linear_llh_step(H_func, x, obs, std_obs, rho_iter)
        else:  # apply Langevin
            z = _langevin_llh_step(
                H_func, x, obs, std_obs, rho_start, langevin_stepsize, n_langevin_steps
            )

        # prior step
        x = denoiser(z, rho_iter)

        # # XXX uncomment to view evolution of reconstruction
        # if i % 10 == 0:
        #     img = x[[0]].clamp(-1.0, 1.0)
        #     display(img)

    return x


def _linear_llh_step(
    H_func: H_functions, x: Tensor, y: Tensor, std_obs: float, rho: float
):
    singulars = H_func.add_zeros(H_func.singulars().unsqueeze(0))
    Qx_inv_eigvals = 1 / (singulars**2 / std_obs**2 + 1 / rho**2)
    noise = H_func.V(
        torch.sqrt(Qx_inv_eigvals) * torch.randn_like(x).reshape(x.shape[0], -1)
    )
    mu_x = H_func.V(
        Qx_inv_eigvals
        * H_func.Vt(H_func.Ht(y / std_obs**2) + (x.reshape(x.shape[0], -1) / rho**2))
    )
    return (mu_x + noise).reshape(*x.shape)


def _langevin_llh_step(
    H_func: H_functions,
    x: Tensor,
    y: Tensor,
    std_obs: float,
    rho: float,
    gamma: float,
    num_iters: int,
):
    z = x
    for _ in range(num_iters):
        z.requires_grad_()

        data_fit = (H_func.H(z) - y).norm() ** 2 / (2 * std_obs**2)
        data_fit.backward()

        with torch.no_grad():
            z = (
                z
                - (gamma * z.grad - (gamma / rho**2) * (z - x))
                + (2 * gamma) ** (0.5) * torch.randn_like(x)
            )

    return z


def _is_svd_decomposed(H_func: H_functions):
    required_methods = ("V", "Vt", "Ht", "singulars")

    for method in required_methods:
        method_fn = getattr(H_func, method)

        # NOTE: since, we want to check wether operator is defined through SVD decomposition
        # ignore other errors, they better get catch in the algo
        try:
            method_fn(None)
        except NotImplementedError:
            return False
        except:
            pass

    return True
