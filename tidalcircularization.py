from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt

try:
    import emcee
except ImportError as e:
    raise ImportError("This module requires `emcee>=3`. Install with `pip install emcee`.") from e

try:
    import corner
    _HAS_CORNER = True
except ImportError:
    _HAS_CORNER = False


class CircularizationModel:
    """
    Tidal circularization curve from Meibom & Mathieu (2005).
    """
    def __init__(self, alpha: float = 0.35, beta: float = 0.14, gamma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, P: np.ndarray, theta: Sequence[float]) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        Pcut = float(theta[0])
        ret = self.alpha * (1.0 - np.exp(self.beta * (Pcut - P))) ** self.gamma
        mask = P < Pcut
        if np.any(mask):
            ret = ret.copy()
            ret[mask] = 0.0
        return ret


class GaussHermiteLikelihood:
    def __init__(self, model: CircularizationModel, n_points: int = 20, epsilon: float = 1e-10):
        self.model = model
        self.n_points = int(n_points)
        self.epsilon = float(epsilon)
        gh_x, gh_w = hermgauss(self.n_points)
        self._gh_points = gh_x
        self._gh_weights = gh_w / np.sqrt(np.pi)

    def log_likelihood(
        self,
        theta: Sequence[float],
        x: np.ndarray,
        y: np.ndarray,
        sigma_x: np.ndarray,
        sigma_y: np.ndarray,
    ) -> float:
        gh_points = self._gh_points
        gh_weights = self._gh_weights

        ll = 0.0
        for i in range(len(x)):
            gh_xi = np.sqrt(2.0) * sigma_x[i] * gh_points + x[i]
            resid = (y[i] - self.model(gh_xi, theta)) / sigma_y[i]
            like_y = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_y[i])) * np.exp(-0.5 * resid**2)
            integral = np.sum(gh_weights * like_y)
            ll += np.log(integral + self.epsilon)
        return float(ll)


class UniformPrior:
    def __init__(self, low: float = 0.0, high: float = 20.0):
        self.low = float(low)
        self.high = float(high)

    def log_prior(self, theta: Sequence[float]) -> float:
        pcut = float(theta[0])
        if self.low < pcut < self.high:
            return 0.0
        return -np.inf


class Posterior:
    def __init__(self, prior: UniformPrior, likelihood: GaussHermiteLikelihood):
        self.prior = prior
        self.likelihood = likelihood

    def __call__(
        self,
        theta: Sequence[float],
        x: np.ndarray,
        y: np.ndarray,
        sigma_x: np.ndarray,
        sigma_y: np.ndarray,
    ) -> float:
        lp = self.prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.likelihood.log_likelihood(theta, x, y, sigma_x, sigma_y)
        lpost = lp + ll
        if not np.isfinite(lpost):
            return -np.inf
        return float(lpost)


@dataclass
class MCMCConfig:
    guess0: np.ndarray = field(default_factory=lambda: np.array([10.0]))   # initial center
    nwalkers: int = 20
    nsamples: int = 20_000
    ncores: Optional[int] = None            # None = no multiprocessing Pool
    nfac: np.ndarray = field(default_factory=lambda: np.array([3.0]))      # width around initial guess
    random_seed: Optional[int] = None
    nburn: int = 500                        # burn-in before thinning

    def init_positions(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        ndim = self.guess0.size
        pos = self.guess0 + self.nfac * rng.normal(size=(self.nwalkers, ndim))
        return pos


@dataclass
class MCMCResults:
    sampler: emcee.EnsembleSampler
    samples: np.ndarray                # [N, ndim] flattened after burn/thin
    names: Sequence[str]
    thin_by: int
    nburn: int
    summary: Dict[str, Tuple[float, float, float]]  # name -> (median, +1σ, -1σ)
    final_positions: np.ndarray        # [nwalkers, ndim]
    final_log_probs: np.ndarray        # [nwalkers]

    def print_results(self) -> None:
        print("emcee results with 1-sigma uncertainties")
        for name, (med, plus, minus) in self.summary.items():
            print(f"{name} = {med:.4f} +{plus:.4f} -{minus:.4f}")

    def plot_chains(self, names: Optional[Sequence[str]] = None, figsize: Tuple[int, int] = (8, 4)):
        """Trace plots using get_chain() (emcee v3)."""
        chain = self.sampler.get_chain()  # shape [nsteps, nwalkers, ndim]
        nsteps, nwalkers, ndim = chain.shape
        if names is None or len(names) != ndim:
            names = [f"v{i}" for i in range(ndim)]

        fig, axes = plt.subplots(ndim, 1, figsize=(figsize[0], 2 * ndim), sharex=True)
        if ndim == 1:
            axes = [axes]
        axes[0].set_title("Chains")
        xplot = np.arange(nsteps)

        for i in range(ndim):
            for w in range(nwalkers):
                axes[i].plot(xplot[: self.nburn], chain[: self.nburn, w, i], alpha=0.4, lw=0.7, zorder=1)
                axes[i].plot(xplot[self.nburn :], chain[self.nburn :, w, i], alpha=0.6, lw=0.7, zorder=1)
            axes[i].set_ylabel(names[i])
        axes[-1].set_xlabel("step")
        return fig, axes

    def plot_corner(
        self,
        names: Optional[Sequence[str]] = None,
        quantiles: Sequence[float] = (0.16, 0.5, 0.84),
        figsize: Tuple[int, int] = (10, 10),
        y_label: Optional[str] = None,
    ):
        if not _HAS_CORNER:
            raise RuntimeError("corner is not installed. `pip install corner` to enable this plot.")
        ndim = self.samples.shape[-1]
        if names is None or len(names) != ndim:
            names = [f"v{i}" for i in range(ndim)]

        fig = plt.figure(figsize=figsize)
        corner.corner(self.samples, labels=names, quantiles=quantiles, fig=fig)

    def plot_model_draws(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma_x: np.ndarray,
        sigma_y: np.ndarray,
        model: CircularizationModel,
        labels: Sequence[str] = ("P", "e"),
        ndraw: int = 20,
    ):
        fig, ax = plt.subplots()
        ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, fmt=".", capsize=3)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.0)
        xvals = 10.0 ** np.linspace(np.log10(float(np.min(x))), np.log10(float(np.max(x))), 1000)

        if ndraw > 0:
            rng = np.random.default_rng()
            idx = rng.integers(0, len(self.samples), size=ndraw)
            for i in idx:
                theta = self.samples[i, :]
                ax.plot(xvals, model(xvals, theta), alpha=0.4, lw=0.7, zorder=1)
        return fig, ax

    def pcirc_distribution(self, model: CircularizationModel, y_target: float = 0.01,
                           P_min: float = 6.0, P_max: float = 50.0, ngrid: int = 1000) -> np.ndarray:
        xvals = 10.0 ** np.linspace(np.log10(P_min), np.log10(P_max), ngrid)
        pcirc = []
        for theta in self.samples:
            y_model = model(xvals, theta)
            pcirc.append(np.interp(y_target, y_model, xvals))
        return np.asarray(pcirc)


DEFAULT_NAMES = ("Pcut",)

def _summarize(samples: np.ndarray, names: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    """Return {name: (median, +1σ, -1σ)}."""
    q = np.percentile(samples, [16, 50, 84], axis=0)
    # shape -> (ndim, 3) as (median, +, -)
    theta = np.array(list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*q))))
    out = {}
    for i, name in enumerate(names):
        out[name] = (float(theta[i, 0]), float(theta[i, 1]), float(theta[i, 2]))
    return out


def run_mcmc(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    prior: Optional[UniformPrior] = None,
    model: Optional[CircularizationModel] = None,
    config: Optional[MCMCConfig] = None,
    names: Sequence[str] = DEFAULT_NAMES,
) -> MCMCResults:
    """Run emcee on the circularization model."""
    model = model or CircularizationModel()
    prior = prior or UniformPrior(0.0, 20.0)
    like = GaussHermiteLikelihood(model)
    post = Posterior(prior, like)

    cfg = config or MCMCConfig()
    pos0 = cfg.init_positions()
    ndim = pos0.shape[1]

    pool = None
    try:
        if cfg.ncores and cfg.ncores > 1:
            import multiprocessing as mp
            pool = mp.get_context("fork").Pool(processes=cfg.ncores)
        sampler = emcee.EnsembleSampler(cfg.nwalkers, ndim, post, args=(x, y, sigma_x, sigma_y), pool=pool)
        sampler.run_mcmc(pos0, cfg.nsamples, progress=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    log_probs = sampler.get_log_prob()        # [nsteps, nwalkers]
    final_log_probs = log_probs[-1, :]
    chain = sampler.get_chain()               # [nsteps, nwalkers, ndim]
    final_positions = chain[-1, :, :]

    try:
        tau = sampler.get_autocorr_time()
        thin_by = int(np.clip(np.mean(tau), 1, None))
        print("Autocorrelation time:", tau)
        print("Thinning interval:", thin_by)
    except Exception as e:
        thin_by = 1
        print(f"Warning: could not estimate autocorrelation time ({e}). Using thin_by=1.")

    flat_samples = sampler.get_chain(discard=cfg.nburn, thin=thin_by, flat=True)
    if len(names) != flat_samples.shape[-1]:
        names = [f"v{i}" for i in range(flat_samples.shape[-1])]

    summary = _summarize(flat_samples, names)

    return MCMCResults(
        sampler=sampler,
        samples=flat_samples,
        names=names,
        thin_by=thin_by,
        nburn=cfg.nburn,
        summary=summary,
        final_positions=final_positions,
        final_log_probs=final_log_probs,
    )

def fit(
    df: pd.DataFrame,
    per_col: str = "Per",
    e_col: str = "e",
    e_per_col: str = "e_Per",
    e_e_col: str = "e_e",
    guess0: Sequence[float] = (10.0,),
    nfac: Sequence[float] = (3.0,),
    nwalkers: int = 20,
    nsamples: int = 20_000,
    nburn: int = 500,
    ncores: Optional[int] = None,
    prior_bounds: Tuple[float, float] = (0.0, 20.0),
    random_seed: Optional[int] = None,
    names: Sequence[str] = DEFAULT_NAMES,
) -> MCMCResults:
    """
    Run the full analysis from a DataFrame with your original column names by default.
    """
    x = np.asarray(df[per_col], dtype=float)
    y = np.asarray(df[e_col], dtype=float)
    sigma_x = np.asarray(df[e_per_col], dtype=float)
    sigma_y = np.asarray(df[e_e_col], dtype=float)

    cfg = MCMCConfig(
        guess0=np.asarray(guess0, dtype=float),
        nwalkers=int(nwalkers),
        nsamples=int(nsamples),
        ncores=ncores,
        nfac=np.asarray(nfac, dtype=float),
        random_seed=random_seed,
        nburn=int(nburn),
    )
    prior = UniformPrior(*prior_bounds)
    return run_mcmc(x, y, sigma_x, sigma_y, prior=prior, config=cfg, names=names)
