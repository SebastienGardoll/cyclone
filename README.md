# Cyclone Detector

## Requirements



## Direct dependencies

The tested version in parenthesis.

- Python 3.7 (3.7.4)
- h5py (2.10.0)
- keras (2.2.4)
- keras-tuner (1.0.1)
- numpy (1.18.1)
- pandas (1.0.1)
- scikit-learn (0.22.1)
- tensorboard (2.0.0)
- tensorflow (2.0.0)


## Dependencies installation script

### Conda

```bash
YOUR_ENV_NAME='gpu_cyclone'
conda create -n ${YOUR_ENV_NAME} python=3.7  # Wait.
conda install -n ${YOUR_ENV_NAME} h5py numpy pandas scikit-learn tensorboard tensorflow'>=2'  # Wait.
source activate ${YOUR_ENV_NAME}
```

### Pip

```bash
YOUR_ENV_NAME='gpu_cyclone'
source activate ${YOUR_ENV_NAME}
pip install -U keras-tuner
```
