# Determining the Tidal Circularization Period for Open Clusters

*Authors: Ritvik Sai Narayan and Aaron Geller*

This package provides a clean, class-based Python interface for fitting tidal circularization models to stellar binary data using an **MCMC** algorithm with [emcee](https://emcee.readthedocs.io/).  

It implements the functional form from *Meibom & Mathieu (2005)* with Gaussian–Hermite quadrature to account for measurement errors in both period and eccentricity.

---

## Installation

Clone your repo and install dependencies:

```bash
git clone https://github.com/ritviksainarayan/tidalcircularization.git
cd tidalcircularization
pip install -r requirements.txt
```

Dependencies:
- `numpy`
- `pandas`
- `matplotlib`
- `emcee>=3`
- `corner` *(optional, for corner plots)*

---

## Quick Start

```python
import pandas as pd
import tidalcircularization as tc

# Load your dataframe with columns:
#   Per     = orbital period
#   e       = eccentricity
#   e_Per   = error in period
#   e_e     = error in eccentricity
df = pd.read_csv("orbits.csv")

# Run MCMC fit (with defaults)
results = tc.fit(df)

# Print parameter summary
results.print_results()

# Corner plot
results.plot_corner()

# Chain trace plots
results.plot_chains()

# Posterior model draws over data
results.plot_model_draws(
    df['Per'], df['e'], df['e_Per'], df['e_e'], tc.CircularizationModel()
)

# Compute Pcirc distribution (eccentricity = 0.01)
pcirc = results.pcirc_distribution(tc.CircularizationModel())
```

---

## API Overview

### Core Classes
- **`CircularizationModel`**
  - Implements the eccentricity vs. period model.
- **`GaussHermiteLikelihood`**
  - Likelihood calculation with quadrature integration.
- **`UniformPrior`**
  - Uniform prior for parameters.
- **`Posterior`**
  - Combines prior and likelihood.
- **`MCMCConfig`**
  - Configuration for sampler (walkers, steps, etc.).
- **`MCMCResults`**
  - Stores chains, samples, plots, summaries, and derived quantities.

### High-level Functions
- **`fit(df, ...)`**
  - One-liner: run full MCMC given a DataFrame.

- **`run_mcmc(x, y, sigma_x, sigma_y, ...)`**
  - Lower-level: run MCMC with explicit arrays.

---

## Configurable Parameters

```python
results = tc.fit(
    df,
    guess0=(10.0),        # initial guess for Pcut
    nfac=(3.0),           # scatter around initial guess
    nwalkers=20,
    nsamples=20_000,
    nburn=500,
    ncores=6,              # multiprocessing cores
    prior_bounds=(0, 20),  # uniform prior bounds for Pcut
    random_seed=42
)
```

---

## Outputs

- **`results.summary`** → dict `{param: (median, +σ, -σ)}`
- **`results.samples`** → posterior draws after burn-in & thinning
- **`results.plot_chains()`** → trace plot
- **`results.plot_corner()`** → posterior correlations (requires `corner`)
- **`results.plot_model_draws(...)`** → posterior model overlays
- **`results.pcirc_distribution(...)`** → array of Pcirc values


## Reference

- **Meibom, S., & Mathieu, R. D. (2005).** "A Study of Tidal Circularization in the Solar-Type Spectroscopic Binary Population of the Open Cluster M35." *ApJ*, 620, 970.

---
