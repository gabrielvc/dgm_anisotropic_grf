# dgm_anisotropic_grf
Repository for the paper "Predictive posterior sampling from non-stationnary Gaussian process priors via Diffusion models with application to climate data."


## Python

First of all, you must go into the datasources repository and install it.
```
cd python/datasources
pip install .
```

### Training the model

To train the model, do the following:

```
cd python/diff_post_gauss
python train_diffusion_script.py --preset YOUR_CONFIGURATION_PRESET
```

You can use the configuration preset for one of the models, as for example `anisotropic_prior_diffusion.yaml`. Note that you will need to adapth the paths, namely for the datasource configuration.

### Generation from DGM

### Classifier test

### Max SW 

### Posterior sampling


