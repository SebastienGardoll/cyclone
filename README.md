# Cyclone Detector

## Requirements

- NXTensor (last version)

## Direct dependencies

The tested version in parenthesis.

- Python 3.7 (3.7.4)
- H5py (2.10.0)
- Keras (2.2.4)
- Keras-tuner (1.0.1)
- Numpy (1.18.1)
- Pandas (1.0.1)
- Scikit-learn (0.22.1)
- Tensorboard (2.0.0)
- Tensorflow (2.0.0)


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
