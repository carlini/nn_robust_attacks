## Corresponding code to the paper <i>"Towards Evaluating the Robustness of Neural Networks"</i> by Nicholas Carlini and David Wagner, at IEEE Symposium on Security & Privacy, 2017.

Implementations of the three attack algorithms in Tensorflow. It runs correctly
on Python 3 (and probably Python 2 without many changes).

### To evaluate the robustness of a neural network:
* Create a model class with apredict method that will run the prediction network *without softmax*.
* The model should have variables:
    - model.image_size: size of the image (e.g., 28 for MNIST, 32 for CIFAR)
    - model.num_channels: 1 for greyscale, 3 for color images
    - model.num_labels: total number of valid labels (e.g., 10 for MNIST/CIFAR)

### Run the attacks with

```python
from robust_attacks import CarliniL2
CarliniL2(sess, model).attack(inputs, targets)
```

#### Note:
* <i>inputs</i> are a (batch x height x width x channels) tensor
* <i>targets</i> are a (batch x classes) tensor.
* The L2 attack supports a batch_size paramater to run attacks in parallel.
* Each attack has many tunable hyper-paramaters.
* All are intuitive and strictly increase attack efficacy in one direction and are more efficient in the other direction.

### The following steps should be sufficient to get these attacks up and running on most Linux-based systems.

```shell
sudo apt-get install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

#### To create the MNIST/CIFAR models:

```shell
python3 train_models.py
```

#### To download the inception model:

```shell
python3 setup_inception.py
```

#### And finally to test the attacks

```shell
python3 test_attack.py
```

This code is provided under the BSD 2-Clause, Copyright 2016 to Nicholas Carlini.
