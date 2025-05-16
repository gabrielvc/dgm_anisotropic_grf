from tqdm import tqdm
from typing import Tuple, Callable

import torch
from torch import Tensor
from posterior_samplers.diffusion_utils import (
    EpsilonNet,
    ddim_step,
    bridge_kernel_statistics,
    ddim_step,
    sample_bridge_kernel,
)
from utils.utils import display
from utils.im_invp_utils import InverseProblem


def mgdm(
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    epsilon_net: EpsilonNet,
    n_reps: int,
    lr_fn: float,
    gradient_steps_fn: int,
    n_denoising_steps: int,
    eta: float = 1.0,
    tau_sampling: int = "mix",
    threshold: int = 70,
    min_tau: int = 10,
    display_im: bool = True,
) -> Tensor:
    timesteps = epsilon_net.timesteps
    log_pot = inverse_problem.log_pot
    n_steps = len(timesteps)

    gradsteps_fn = get_gradientsteps_fn(gradient_steps_fn, n_steps)
    lr_fn = get_lr_fn(lr_fn, n_steps)
    tau_sample_fn = get_tau_sampling(
        type=tau_sampling, min_tau=min_tau, threshold=threshold
    )

    x_t = initial_noise
    x_0 = epsilon_net.predict_x0(x_t, epsilon_net.timesteps[-1])

    flipd_timesteps = timesteps.flip(0)
    pbar = tqdm(
        enumerate(
            zip(flipd_timesteps[2:-1], flipd_timesteps[1:-2], flipd_timesteps[:-2])
        )
    )

    for idx, (t_prev, t, t_next) in pbar:

        tau = tau_sample_fn(t_prev, idx)
        ngrad_steps = gradsteps_fn(idx)
        lr = lr_fn(idx)

        x_tnext = x_t
        x_t = sample_bridge_kernel(x_tnext, x_0, epsilon_net, t_next, t, 0, eta=eta)

        log_pot_fn = lambda x: log_pot(epsilon_net.predict_x0(x, tau))

        for _ in range(n_reps):
            x_0, x_t = gibbs_step(
                # fmt: off
                x_0, x_t, x_tnext,
                tau, t, t_next,
                # fmt: on
                epsilon_net,
                log_pot_fn,
                lr,
                ngrad_steps,
                n_denoising_steps,
                eta=eta,
            )

        # # XXX uncomment to see the evolution of x_0
        # if display_im and idx % 10 == 0:
        #     for sample in x_0:
        #         if x_0.shape[-1] == 256:
        #             display(sample.clamp(-1, 1), title=f"tau: {tau}")
        #         elif x_0.shape[-1] == 64:
        #             display(
        #                 epsilon_net.decode(sample.unsqueeze(0)).clamp(-1, 1),
        #                 title=f"tau: {tau}",
        #             )

    return x_0


def gibbs_step(
    # fmt: off
    x_0, x_t, x_tnext,
    tau, t, t_next,
    # fmt: on
    epsilon_net: EpsilonNet,
    log_pot: Callable[[Tensor], Tensor],
    lr: float,
    n_gradient_steps: int,
    n_denoising_steps: int,
    eta: float,
) -> Tuple[Tensor, Tensor]:
    # potential
    x_tau = _pot_update_stochastic(
        # fmt: off
        x_0, x_t,
        tau, t,
        # fmt: on
        log_pot,
        epsilon_net,
        lr,
        n_gradient_steps,
    )

    # denoising step
    x_0 = _denoising_step(0, tau, x_tau, epsilon_net, n_denoising_steps, eta=eta)

    # noising step
    x_t = sample_bridge_kernel(x_tnext, x_tau, epsilon_net, t_next, t, tau)

    return x_0, x_t


# ------ utils
def _pot_update_stochastic(
    # fmt: off
    x_0, x_t,
    tau, t,
    # fmt: on
    log_pot: Callable[[Tensor], Tensor],
    epsilon_net: EpsilonNet,
    lr: float,
    n_gradient_steps: int,
) -> Tensor:
    """
    Return
    ------
    x_tau
    """
    n = x_t.shape[0]
    mean_prior, std_prior = bridge_kernel_statistics(
        x_t, x_0, epsilon_net, t, tau, 0, eta=1.0
    )
    log_std_prior = std_prior.log()

    vmean = mean_prior.clone()
    vlog_std = log_std_prior * torch.ones_like(x_t)
    vmean.requires_grad_(), vlog_std.requires_grad_()
    optim = torch.optim.Adam(params=[vmean, vlog_std], lr=lr)

    for _ in range(n_gradient_steps):
        optim.zero_grad()
        vsample = vmean + vlog_std.exp() * torch.randn_like(mean_prior)

        loss = (
            -log_pot(vsample)
            + 0.5 * (((vsample - mean_prior) / std_prior) ** 2).sum()
            - vlog_std.sum()
        )
        loss.backward()

        optim.step()

    with torch.no_grad():
        x_tau = vmean + vlog_std.exp() * torch.randn_like(x_t)

    return x_tau


def _denoising_step(
    s, tau, x_tau: Tensor, epsilon_net: EpsilonNet, n_denoising_steps: int, eta: float
) -> Tensor:
    if isinstance(s, int):
        s = torch.tensor(s, dtype=torch.int32)
    if isinstance(tau, int):
        tau = torch.tensor(tau, dtype=torch.int32)

    s = s.clamp(0, None)
    n_steps = min(n_denoising_steps, tau - s + 1)

    if n_steps < 2:
        return epsilon_net.predict_x0(x_tau, tau)

    arr_times = torch.linspace(s, tau, n_steps, dtype=torch.int32).flip(0)
    sequence_k_k_prev = zip(arr_times[:-1], arr_times[1:])

    x_k = x_tau
    for k, k_prev in sequence_k_k_prev:
        x_k = ddim_step(x_k, epsilon_net, k, k_prev, eta=eta)

    return x_k


def get_tau_sampling(type="uniform", **kwargs):
    if type == "uniform":
        min_tau = kwargs.get("min_tau")
        return lambda t, idx: torch.randint(min_tau, t, (1,))[0]
    elif type == "zero":
        return lambda t, idx: torch.randint(1, max(1, t // 5), (1,))[0]
    elif type == "mix":
        threshold = kwargs.get("threshold")
        min_tau = kwargs.get("min_tau")
        return lambda t, idx: (
            torch.randint(min_tau, t, (1,))[0] if idx < threshold else t
        )
    elif type == "deterministic":
        return lambda t, idx: t
    elif type == "mix-det":
        threshold = kwargs.get("threshold")
        return lambda t, idx: (75 * t) // 100 if idx < threshold else t
    else:
        raise ValueError("time sampler not implemented")


def get_gradientsteps_fn(gradientsteps_fn, n_steps):

    def fn(i):
        for rule in gradientsteps_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    return fn


def get_lr_fn(lr_fn, n_steps):

    def fn(i):
        for rule in lr_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    return fn
