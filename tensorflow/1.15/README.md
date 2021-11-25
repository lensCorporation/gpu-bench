<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" align="right" width="100"/>


# GPU-Bench

> GPU Benchmarking on TensorFlow 1.15

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" width="25"/> Tensorflow release
Currently this repo is compatible with Tensorflow 1.15.

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Breezeicons-actions-22-run-build-install-root.svg/640px-Breezeicons-actions-22-run-build-install-root.svg.png" width="25"/> Installation
Installation covers deploying a conda environment with TF 1.15 installed.
```
  1. conda create -n "tf1.15" python=3.8
  2. pip install nvidia-pyindex
  3. pip install nvidia-tensorflow[horovod]
  4. pip install imageio scipy
```

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Emoji_u1f3c3_1f3fd.svg/640px-Emoji_u1f3c3_1f3fd.svg.png" width="25"/> Running
```
   python train.py config.py
```

<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License
Distributed under the Apache License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Debayan Deb - debayan@lenscorp.ai
Azhan Mohammed - mohd.azhan@lenscorp.ai
Aishvary Pratap - aishvary@lenscorp.ai
