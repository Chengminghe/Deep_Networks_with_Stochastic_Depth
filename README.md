# Deep Networks with Stochastic Depth
This repo reproduces the work in https://arxiv.org/abs/1603.09382. 
The implementation is in python with tensorflow2.4. Relevant jupyter notebooks are in the root directory and utility files are in the `utils` directory. 
The report includes all the implementation details and results.

```
./
├── Accuracies
│   └── acc
├── Logs
│   ├── ResNet110_cifar10.npy
│   ├── ResNet110_cifar100.npy
│   ├── ResNet110_cifar10_500.npy
│   ├── ResNet110_food101.npy
│   ├── ResNet152_svhn.npy
│   ├── ResNet302_cifar10.npy
│   ├── StochasticNet110_cifar10.npy
│   ├── StochasticNet110_cifar100.npy
│   ├── StochasticNet110_cifar10_2000.2.npy
│   ├── StochasticNet110_cifar10_2000.8.npy
│   ├── StochasticNet110_cifar10_200_uniform0.2.npy
│   ├── StochasticNet110_cifar10_200_uniform0.5.npy
│   ├── StochasticNet110_cifar10_200_uniform0.8.npy
│   ├── StochasticNet110_cifar10_500.npy
│   ├── StochasticNet110_food101.npy
│   ├── StochasticNet110_svhn_0.5.npy
│   ├── StochasticNet110_svhn_0.6.npy
│   ├── StochasticNet110_svhn_0.7.npy
│   ├── StochasticNet110_svhn_0.8.npy
│   ├── StochasticNet110_svhn_0.9.npy
│   ├── StochasticNet110_svhn_1.npy
│   ├── StochasticNet152_food101.npy
│   ├── StochasticNet152_food101_1.npy
│   ├── StochasticNet152_svhn.npy
│   ├── StochasticNet20_0.50.npy
│   ├── StochasticNet20_0.60.npy
│   ├── StochasticNet20_0.70.npy
│   ├── StochasticNet20_0.80.npy
│   ├── StochasticNet20_0.90.npy
│   ├── StochasticNet20_1.00.npy
│   ├── StochasticNet302_cifar10.npy
│   ├── StochasticNet38_0.50.npy
│   ├── StochasticNet38_0.60.npy
│   ├── StochasticNet38_0.70.npy
│   ├── StochasticNet38_0.80.npy
│   ├── StochasticNet38_0.90.npy
│   ├── StochasticNet38_1.00.npy
│   ├── StochasticNet56_0.50.npy
│   ├── StochasticNet56_0.60.npy
│   ├── StochasticNet56_0.70.npy
│   ├── StochasticNet56_0.80.npy
│   ├── StochasticNet56_0.90.npy
│   ├── StochasticNet56_1.00.npy
│   ├── StochasticNet74_0.50.npy
│   ├── StochasticNet74_0.60.npy
│   ├── StochasticNet74_0.70.npy
│   ├── StochasticNet74_0.80.npy
│   ├── StochasticNet74_0.90.npy
│   ├── StochasticNet74_1.00.npy
│   ├── StochasticNet92_0.50.npy
│   ├── StochasticNet92_0.60.npy
│   ├── StochasticNet92_0.70.npy
│   ├── StochasticNet92_0.80.npy
│   ├── StochasticNet92_0.90.npy
│   ├── StochasticNet92_1.00.npy
│   ├── StochasticNet92_svhn_0.5.npy
│   ├── StochasticNet92_svhn_0.6.npy
│   ├── StochasticNet92_svhn_0.7.npy
│   ├── StochasticNet92_svhn_0.8.npy
│   ├── StochasticNet92_svhn_0.9.npy
│   ├── StochasticNet92_svhn_1.npy
│   ├── resnet_food101.txt
│   ├── stochastic152_food101.txt
│   ├── stochastic152_food101_1.txt
│   └── stochastic_food101.txt
├── Models
│   ├── ResNet110_cifar10_500
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── ResNet110_cifar10_500_final
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_2000.2
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_2000.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_final0.2
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_final0.5
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_final0.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_uniform0.2
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_uniform0.5
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_uniform0.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_uniform_final0.2
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_uniform_final0.5
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_200_uniform_final0.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_500
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_cifar10_500_final
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_0.5
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_0.6
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_0.7
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_0.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_0.9
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_1
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_final_0.5
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_final_0.6
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_final_0.7
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_final_0.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_final_0.9
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet110_svhn_final_1
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_0.5
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_0.6
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_0.7
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_0.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_0.9
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_1
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_final_0.5
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_final_0.6
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_final_0.7
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_final_0.8
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── StochasticNet92_svhn_final_0.9
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   └── StochasticNet92_svhn_final_1
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
├── README.md
├── ResNet110_cifar10.ipynb
├── ResNet110_cifar100.ipynb
├── ResNet152_svhn.ipynb
├── ResNet302_cifar10.ipynb
├── ResNet_cifar10.ipynb
├── ResNet_food101.ipynb
├── Stochastic302_cifar10.ipynb
├── StochasticNet.ipynb
├── StochasticNet110_cifar10.ipynb
├── StochasticNet110_cifar100.ipynb
├── StochasticNet152_svhn.ipynb
├── Stochastic_food101.ipynb
├── Stochastic_withtensorboard.ipynb
├── Stochasticnet_withtensorboard.ipynb
├── plots.ipynb
├── plots2.ipynb
└── utils
    ├── Resnet_cifar10_train.py
    ├── StochasticNet_svhn_train.py
    ├── Stochastic_cifar10_train.py
    ├── Stochastic_food101.py
    ├── neuralnets
    │   ├── ResNet110.py
    │   ├── ResNet152.py
    │   ├── ResNet302.py
    │   ├── StochasticNet.py
    │   ├── StochasticNet110.py
    │   ├── StochasticNet152.py
    │   ├── StochasticNet302.py
    │   └── __pycache__
    │       ├── ResNet110.cpython-36.pyc
    │       ├── ResNet110.cpython-36.pyc.orig
    │       ├── ResNet110.cpython-38.pyc
    │       ├── ResNet1202.cpython-36.pyc
    │       ├── ResNet152.cpython-36.pyc
    │       ├── ResNet156.cpython-36.pyc
    │       ├── ResNet302.cpython-36.pyc
    │       ├── StochasticNet.cpython-36.pyc
    │       ├── StochasticNet110.cpython-36.pyc
    │       ├── StochasticNet110.cpython-36.pyc.orig
    │       ├── StochasticNet110.cpython-38.pyc
    │       ├── StochasticNet152.cpython-36.pyc
    │       ├── StochasticNet152.cpython-36.pyc.orig
    │       └── StochasticNet302.cpython-36.pyc
    ├── resnet_food101.py
    ├── resnetblock.py
    ├── stochasticblock.py
    └── trainer.py

84 directories, 229 files
```
