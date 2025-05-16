# %%
import jax
import jax.numpy as jnp
from numpyro.distributions import (
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
    Categorical,
)
import os
from jax.random import PRNGKey, split
from jax.nn import softmax, logsumexp
from jax import grad, vmap
import matplotlib.pyplot as plt
from jax.scipy.linalg import inv, cho_solve, cho_factor
from jax.scipy.stats import multivariate_normal, norm
from jax.lax import cond
from jax.tree_util import Partial as partial
from jax.scipy.integrate import trapezoid


def gmm_1d(key, std=1.0, n_mixtures=25):
    key_means, key_weights = split(key)
    means = 30 * (2 * jax.random.uniform(key_means, (n_mixtures,)) - 1)
    norm = Normal(means, jnp.repeat(jnp.array(std), 25))
    weights = jax.random.uniform(key_weights, (n_mixtures,))
    weights = weights / weights.sum()
    return MixtureSameFamily(Categorical(weights), norm)


def get_posterior(obs, A, noise_std, prior):
    modified_means = []
    modified_vars = []
    log_weights = []

    for loc, var, log_weight in zip(
        prior.component_distribution.mean,
        prior.component_distribution.variance,
        prior.mixing_distribution.logits,
    ):
        # Compute the posterior for each component
        new_dist = gaussian_posterior(
            obs,
            A,
            noise_std,
            loc,
            var,
        )
        modified_means.append(new_dist.loc)
        modified_vars.append(new_dist.variance)

        # Compute prior log-probability and log constant
        y_logprob = norm.logpdf(
            obs, loc=A * loc, scale=jnp.sqrt(noise_std**2 + (A**2) * var)
        )
        log_weights.append(log_weight + y_logprob)

    log_weights = jnp.array(log_weights)
    modified_means = jnp.array(modified_means).flatten()
    modified_vars = jnp.array(modified_vars).flatten()

    posterior_weights = softmax(log_weights, axis=0).flatten()
    component_posteriors = Normal(modified_means, modified_vars)
    return MixtureSameFamily(
        mixing_distribution=Categorical(probs=posterior_weights),
        component_distribution=component_posteriors,
    )


def gaussian_posterior(obs, A, std, prior_loc, prior_var):
    obs_var = std**2
    posterior_var = obs_var * prior_var / (obs_var + (A**2) * prior_var)
    posterior_loc = obs_var * prior_loc + prior_var * A * obs
    posterior_loc = posterior_loc / (obs_var + (A**2) * prior_var)
    return Normal(posterior_loc, posterior_var)


def generate_invp(key, A, obs_std, prior):
    key_sample, key_noise = split(key)
    sample = prior.sample(key_sample, (1,))
    return A * sample + obs_std * jax.random.normal(key_noise, shape=sample.shape)


def true_logpot(obs, A, obs_std, mixture):
    mixture_loc = mixture.component_distribution.mean
    mixture_var = mixture.component_distribution.variance
    log_weights = mixture.mixing_distribution.logits
    n_mixture = len(mixture_loc)
    y_logprob = norm.logpdf(
        jnp.repeat(obs[None, :], n_mixture),
        A * mixture_loc,
        jnp.sqrt(obs_std**2 + (A**2) * mixture_var),
    )

    return logsumexp(log_weights + y_logprob)


def fwd_mixture(mixture, alphas_cumprod, t):
    means, variances, weights = (
        mixture.component_distribution.mean,
        mixture.component_distribution.variance,
        mixture.mixing_distribution.probs,
    )

    acp_t = alphas_cumprod[t]
    mixture_means = jnp.sqrt(acp_t) * means
    mixture_variances = (1 - acp_t) + acp_t * variances
    norm = Normal(mixture_means, mixture_variances)
    return MixtureSameFamily(Categorical(weights), norm)


def eps_net_mixture(mixture, alphas_cumprod):

    def fn(x, t):
        acp_t = alphas_cumprod[t]
        grad_logprob = grad(
            lambda x: fwd_mixture(mixture, alphas_cumprod, t).log_prob(x).sum()
        )
        return -((1 - acp_t) ** 0.5) * grad_logprob(x)

    return fn


class EpsilonNet:
    def __init__(self, net, alphas_cumprod, timesteps):
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps

    def predict_eps(self, x, t):
        return self.net(x, t)

    def predict_x0(self, x, t):
        acp_t = self.alphas_cumprod[t]
        return (x - (1 - acp_t) ** 0.5 * self.predict_eps(x, t)) / (acp_t**0.5)

    def score(self, x, t):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.predict_eps(x, t) / (1 - acp_t) ** 0.5


def load_gmm_epsilon_net(prior, n_steps):
    timesteps = jnp.linspace(0, 999, n_steps)
    alphas_cumprod = jnp.linspace(0.9999, 0.98, 1000)
    alphas_cumprod = jnp.cumprod(alphas_cumprod, 0).clip(1e-10, 1)
    alphas_cumprod = jnp.concatenate([jnp.array([1.0]), alphas_cumprod])

    epsilon_net = EpsilonNet(
        net=eps_net_mixture(prior, alphas_cumprod),
        alphas_cumprod=alphas_cumprod,
        timesteps=timesteps,
    )

    return epsilon_net


def bw_transition(x, s, t, eps_net, prior_s):
    acp_ratio = eps_net.alphas_cumprod[t] / eps_net.alphas_cumprod[s]
    A = jnp.sqrt(acp_ratio)
    noise_std = jnp.sqrt(1 - acp_ratio)
    return get_posterior(x, A, noise_std, prior_s)


def rectangle_rule_logsumexp(log_f, a, b, n):
    dx = (b - a) / n
    midpoints = a + (jnp.arange(n) + 0.5) * dx
    log_f_values = log_f(midpoints)
    return logsumexp(log_f_values) + jnp.log(dx)


def compute_true_logpot(x, t, obs, A, obs_std, eps_net, prior):
    bw = bw_transition(x, 0, t, eps_net, prior)
    return true_logpot(obs, A, obs_std, bw)


def compute_mdps_logpot(x, s, t, obs, A, obs_std, eps_net, prior, n_points=5000):
    prior_s = fwd_mixture(prior, eps_net.alphas_cumprod, s)
    bw_st = bw_transition(x, s, t, eps_net, prior_s)
    dps_logpot = lambda x: norm.logpdf(obs, A * eps_net.predict_x0(x, s), obs_std)
    logjoint_fn = lambda x: dps_logpot(x) + bw_st.log_prob(x)
    return rectangle_rule_logsumexp(logjoint_fn, a=-40, b=40, n=n_points)


def compute_logpot_error(key, s, t, obs, A, obs_std, eps_net, prior, n_samples=1000):
    logpot_t = partial(
        compute_true_logpot,
        t=t,
        obs=obs,
        A=A,
        obs_std=obs_std,
        eps_net=eps_net,
        prior=prior,
    )
    logpot_ts = partial(
        compute_mdps_logpot,
        s=s,
        t=t,
        obs=obs,
        A=A,
        obs_std=obs_std,
        eps_net=eps_net,
        prior=prior,
    )

    key_prior, _ = split(key)
    posterior = get_posterior(obs, A, obs_std, prior)
    posterior_t = fwd_mixture(posterior, eps_net.alphas_cumprod, t)

    # prior_min = prior.component_distribution.loc.min()
    # prior_max = prior.component_distribution.loc.max()
    # eps = 5

    # xs = (prior_min - eps) + (prior_max + eps) * jax.random.uniform(
    #     key_unif, shape=(n_samples_unif)
    # )
    # xs = xs.reshape(-1, 1)

    xt = posterior_t.sample(key_prior, (n_samples,))
    # xt = -40 + 40 * jax.random.uniform(key_prior, (n_samples, ))

    logpot_val = vmap(logpot_t)(xt)
    logpot_val_s = vmap(logpot_ts)(xt)

    return jnp.abs(logpot_val - logpot_val_s).mean()
