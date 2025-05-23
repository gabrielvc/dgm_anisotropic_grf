# dgm_anisotropic_grf
Repository for the paper "Predictive posterior sampling from non-stationnary Gaussian process priors via Diffusion models with application to climate data."


## Python

First of all, you must install the required requirements:
```
pip install -r python_requirements.txt
```
Then, you must go into the datasources repository and install it.
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

Once the model is trained, one can generate samples by running the command

```
cd python/diff_post_gauss
python generate_samples.py --preset TRAINED_PRESET --ckpt_path PATH_TO_CHECKPOINT --batch_size BATCH_SIZE --n_samples HOW_MANY_SAMPLES_YOU_WANT --config_sampler CONFIGURATION_FOR_SAMPLER --save_folder WHERE_TO_SAVE --trainer_offset SEED_FOR_GENERATION
```
This will create a subfolder in `WHERE_TO_SAVE` with the name of the sampler from `CONFIGURATION_FOR_SAMPLER` which will then have a subfolder named `raw_data` where a `SEED.pt` file will be created.
One can then reassemble all those `.pt` files into a `.h5`file by running
```
cd python/diff_post_gauss
python join_batch_generated_data_into_h5.py --path PATH_TO_FOLDER_CONTAINING_RAW_DATA --field_name samples
```
This will create a `data.h5̀` file in `PATH_TO_FOLDER_CONTAINING_RAW_DATA` with everything that was inside `PATH_TO_FOLDER_CONTAINING_RAW_DATA/raw_data`.

### Classifier test

Once data is generated and in an `.h5` format, one can perform a classifier 2 sample test.
To do so, run the following script
```
cd python/diff_post_gauss
python classifier_tests.py --ref_data_path PATH_TO_REFERENCE_DATASET --gen_data_path PATH_GEZNERATED_DATA/data.h5 --batch_size BATCH_SIZE --model_type MODEL_TYPE --seed 42 --acc_grads 1 --ckpt_path WHERE_TO_SAVE
```
This will produce mainly tensorboard logs from where the information can be extracted using the `tbparse` python package. An example of such scripts, mirroring one that we used to produce the data is given in `python/diff_post_gauss/extract_info_tensorboard.py`.
The variable `MODEL_TYPE` should be either `resnet18`, `resnet50` or `resnet101`.

### Max SW 

For evaluating the Max sliced wasserstein, we used the `pot`python library. 
An example, mirroring the script that we used to do so is given in `python/diff_post_gauss/evaluate_max_sliced_wasserstein.py`.

### Posterior sampling

For posterior sampling one should go into `python/diff_post_gauss`. Then, one should install the `mgdm`(which is a slight modification of [the original mgdm package](https://github.com/Badr-MOUFAD/mgdm) package by runing `pip install -e .`and finally run
```
cd python/diff_post_gauss
posterior_sampling.py --denoiser_preset DENOISER_PRESET --ckpt_path PATH_TO_CHECKPOINT --sampler_preset WHICH_POSTERIOR_SAMPLER_CONFIG_TO_USE --save_folder=WHERE_TO_SAVE_SAMPLES --seed_offset 42 --inverse_problem_preset INVERSE_PROBLEM_CONFIGURATION_TO_USE
```
Like for the script that generates samples, this script will generate a `WHERE_TO_SAVE_SAMPLES/raw_data` folder containing `.pt`files that can be joined using the join
into an `data.h5`file using `join_batch_generated_data_into_h5.py`.