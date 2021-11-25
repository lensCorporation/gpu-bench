<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" align="right" width="100"/>


# GPU-Bench

> GPU Benchmarking on TensorFlow 1.15

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" width="25"/> Tensorflow release
Currently this repo is compatible with Tensorflow 1.15.

## <img src="https://image.flaticon.com/icons/svg/1/1383.svg" width="25"/> Installation
Installation covers deploying a conda environment with TF 1.15 installed.
```
  1. conda create -n "tf1.15" python=3.8
  2. pip install nvidia-pyindex
  3. pip install nvidia-tensorflow[horovod]
  4. pip install imageio scipy
```

## <img src="https://image.flaticon.com/icons/svg/1/1383.svg" width="25"/> Running
```
   python train.py config.py
```
