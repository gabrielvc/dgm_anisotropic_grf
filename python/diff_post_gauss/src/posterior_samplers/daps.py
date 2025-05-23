import torch
from posterior_samplers.diffusion_utils import EpsilonNet
from torch import nn
import tqdm
import numpy as np
from utils.im_invp_utils import InverseProblem
from typing import Dict
from posterior_samplers.diffusion_utils import VPPrecond, VEPrecond
from sound_model.misc import SoundModelWrapper
from utils.utils import display
from utils.experiments_tools import save_im, rm_files


def daps(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    inverse_problem: InverseProblem,
    lr: float,
    tau: float,
    lr_min_ratio: float,
    lgv_steps: int,
    annealing_scheduler_config: Dict[str, float],
    diffusion_scheduler_config: Dict[str, float],
    debug_plots: bool = False,
):
    """Wrapper over the DAPS [1] implementation.

    Code adapted from https://github.com/zhangbingliang2019/DAPS

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    inverse_problem : Tuple
        observation, degradation operator, and standard deviation of noise.

    epsilon_net: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    References
    ----------
    .. [1] Zhang, Bingliang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song.
        "Improving diffusion inverse problem solving with decoupled noise annealing."
        arXiv preprint arXiv:2407.01521 (2024).
    """
    langevin_config = {
        "lr": lr,
        "tau": tau,
        "lr_min_ratio": lr_min_ratio,
        "num_steps": lgv_steps,
    }
    sampler = DAPS(
        annealing_scheduler_config, diffusion_scheduler_config, langevin_config
    )
    model = DDPM(epsilon_net=epsilon_net)
    initial_noise = (
        torch.randn_like(initial_noise) * annealing_scheduler_config.sigma_max
    )
    samples = sampler.sample(
        model,
        initial_noise,
        operator=inverse_problem.H_func.H,
        measurement=inverse_problem.obs,
        debug_plots=debug_plots,
    )
    return samples


class DAPS(nn.Module):
    """
    Implementation of decoupled annealing posterior sampling.
    """

    def __init__(
        self, annealing_scheduler_config, diffusion_scheduler_config, lgvd_config
    ):
        """
        Initializes the DAPS sampler with the given configurations.

        Parameters:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(
            annealing_scheduler_config, diffusion_scheduler_config
        )
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    def sample(
        self,
        model,
        x_start,
        operator,
        measurement,
        display_im=True,
        debug_plots=False,
        verbose=True,
        **kwargs,
    ):

        pbar = (
            tqdm.trange(self.annealing_scheduler.num_steps)
            if verbose
            else range(self.annealing_scheduler.num_steps)
        )
        xt = x_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            diffusion_scheduler = Scheduler(
                **self.diffusion_scheduler_config, sigma_max=sigma
            )
            sampler = DiffusionSampler(diffusion_scheduler)
            x0hat = sampler.sample(model, xt, SDE=False, verbose=False)

            # 2. langevin dynamics
            x0y = self.lgvd.sample(
                x0hat,
                operator,
                measurement,
                sigma,
                step / self.annealing_scheduler.num_steps,
            )

            # 3. forward diffusion
            xt = (
                x0y
                + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            )

            if step % 10 == 0:
                if display_im:
                    import matplotlib.pyplot as plt
                    plt.imshow(x0y[0, 0].cpu())
                    plt.savefig("test_sample.png")
                    # for sample in x0y:
                    #     if x0y.shape[-1] == 256:
                    #         display(sample.clamp(-1, 1))
                    #     # elif x0y.shape[-1] == 64:
                        #     display(
                        #         epsilon_net.decode(sample.unsqueeze(0)).clamp(-1, 1),
                        #         title=f"tau: {tau}",
                        #     )

        return xt

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
        Checks and updates the configurations for the schedulers.
        """
        # sigma_max of diffusion scheduler change each step
        if "sigma_max" in diffusion_scheduler_config:
            diffusion_scheduler_config.pop("sigma_max")

        # sigma final of annealing scheduler should always be 0
        annealing_scheduler_config["sigma_final"] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, ref):
        """
        Generates a random initial state based on the reference tensor.

        Parameters:
            ref (torch.Tensor): Reference tensor for shape and device.

        Returns:
            torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start


class DiffusionSampler(nn.Module):
    """
    Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver="euler"):
        """
        Initializes the diffusion sampler with the given scheduler and solver.

        Parameters:
            scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
            solver (str): Solver method ('euler').
        """
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, SDE=False, record=False, verbose=False):
        """
        Samples from the diffusion process using the specified model.

        Parameters:
            model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
            x_start (torch.Tensor): Initial state.
            SDE (bool): Whether to use Stochastic Differential Equations.
            record (bool): Whether to record the trajectory.
            verbose (bool): Whether to display progress bar.

        Returns:
            torch.Tensor: The final sampled state.
        """
        if self.solver == "euler":
            return self._euler(model, x_start, SDE, record, verbose)
        else:
            raise NotImplementedError

    def _euler(self, model, x_start, SDE=False, record=False, verbose=False):
        """
        Euler's method for sampling from the diffusion process.
        """
        pbar = (
            tqdm.trange(self.scheduler.num_steps)
            if verbose
            else range(self.scheduler.num_steps)
        )

        x = x_start
        for step in pbar:
            sigma, factor = (
                self.scheduler.sigma_steps[step],
                self.scheduler.factor_steps[step],
            )
            score = model.score(x, sigma)
            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            # record
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)
        return x


class LangevinDynamics(nn.Module):
    """
    Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        """
        Initializes the Langevin dynamics sampler with the given parameters.

        Parameters:
            num_steps (int): Number of steps in the sampling process.
            lr (float): Learning rate.
            tau (float): Noise parameter.
            lr_min_ratio (float): Minimum learning rate ratio.
        """
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio

    def sample(
        self, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False
    ):
        """
        Samples using Langevin dynamics.

        Parameters:
            x0hat (torch.Tensor): Initial state.
            operator (Operator): Operator module.
            measurement (torch.Tensor): Measurement tensor.
            sigma (float): Current sigma value.
            ratio (float): Current step ratio.
            record (bool): Whether to record the trajectory.
            verbose (bool): Whether to display progress bar.

        Returns:
            torch.Tensor: The final sampled state.
        """
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()
            # loss = operator.error(x, measurement).sum() / (2 * self.tau**2)
            loss = (measurement - operator(x)).square().sum() / (2 * self.tau**2)
            loss += ((x - x0hat.detach()) ** 2).sum() / (2 * sigma**2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x)

            # record
            if record:
                self._record(x, epsilon, loss)
        return x.detach()

    def get_lr(self, ratio):
        """
        Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (
            1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))
        ) ** p
        return multiplier * self.lr


class Scheduler(nn.Module):
    """
    Scheduler for diffusion sigma(t) and discretization step size Delta t
    """

    def __init__(
        self,
        num_steps=10,
        sigma_max=100,
        sigma_min=0.01,
        sigma_final=None,
        schedule="linear",
        timestep="poly-7",
    ):
        """
        Initializes the scheduler with the given parameters.

        Parameters:
            num_steps (int): Number of steps in the schedule.
            sigma_max (float): Maximum value of sigma.
            sigma_min (float): Minimum value of sigma.
            sigma_final (float): Final value of sigma, defaults to sigma_min.
            schedule (str): Type of schedule for sigma ('linear' or 'sqrt').
            timestep (str): Type of timestep function ('log' or 'poly-n').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(
            self.timestep, self.sigma_max, self.sigma_min
        )

        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        # factor = 2\dot\sigma(t)\sigma(t)\Delta t
        factor_steps = np.array(
            [
                2
                * sigma_fn(time_steps[i])
                * sigma_derivative_fn(time_steps[i])
                * (time_steps[i] - time_steps[i + 1])
                for i in range(num_steps)
            ]
        )
        self.sigma_steps, self.time_steps, self.factor_steps = (
            sigma_steps,
            time_steps,
            factor_steps,
        )
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        """
        Returns the sigma function, its derivative, and its inverse based on the given schedule.
        """
        if schedule == "sqrt":
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma**2

        elif schedule == "linear":
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
        Returns the time step function based on the given timestep type.
        """
        if timestep == "log":
            get_time_step_fn = (
                lambda r: sigma_max**2 * (sigma_min**2 / sigma_max**2) ** r
            )
        elif timestep.startswith("poly"):
            p = int(timestep.split("-")[1])
            get_time_step_fn = (
                lambda r: (
                    sigma_max ** (1 / p)
                    + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))
                )
                ** p
            )
        else:
            raise NotImplementedError
        return get_time_step_fn


class DiffusionModel(nn.Module):
    """
    A class representing a diffusion model.
    Methods:
        score(x, sigma): Calculates the score function of time-varying noisy distribution:

                \nabla_{x_t}\log p(x_t;\sigma_t)

        tweedie(x, sigma): Calculates the expectation of clean data (x0) given noisy data (xt):

             \mathbb{E}_{x_0 \sim p(x_0 \mid x_t)}[x_0 \mid x_t]
    """

    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Check if either `score` or `tweedie` is overridden
        if (
            self.score.__func__ is DiffusionModel.score
            and self.tweedie.__func__ is DiffusionModel.tweedie
        ):
            raise NotImplementedError(
                "Either `score` or `tweedie` method must be implemented."
            )

    def score(self, x, sigma):
        """
        x       : noisy state at time t, torch.Tensor([B, *data.shape])
        sigma   : noise level at time t, scaler
        """
        d = self.tweedie(x, sigma)
        return (d - x) / sigma**2

    def tweedie(self, x, sigma):
        """
        x       : noisy state at time t, torch.Tensor([B, *data.shape])
        sigma   : noise level at time t, scaler
        """
        return x + self.score(x, sigma) * sigma**2


class DDPM(DiffusionModel):
    """
    DDPM (Diffusion Denoising Probabilistic Model)
    Attributes:
        model (VPPrecond): The neural network used for denoising.

    Methods:
        __init__(self, model_config, device='cuda'): Initializes the DDPM object.
        tweedie(self, x, sigma=2e-3): Applies the DDPM model to denoise the input, using VPE preconditioning from EDM.
    """

    def __init__(self, epsilon_net, requires_grad=False):
        super().__init__()

        # XXX case-by-base for sound model
        # TODO: to be improved later to support native
        if isinstance(epsilon_net.net, SoundModelWrapper):
            self.model = VEPrecond(epsilon_net.net.denoiser, epsilon_net.net.sigmas)
        else:
            self.model = VPPrecond(model=epsilon_net)

        self.model.eval()
        self.model.requires_grad_(requires_grad)

    def tweedie(self, x, sigma=2e-3):
        return self.model(x, torch.as_tensor(sigma))
