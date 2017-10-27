### About

Corresponding code to the paper "Towards Evaluating the Robustness of Neural
Networks" by Nicholas Carlini and David Wagner, at IEEE Symposium on Security &
Privacy, 2017.

Implementations of the three attack algorithms in Tensorflow. It runs correctly
on Python 3 (and probably Python 2 without many changes).

To evaluate the robustness of a neural network, create a model class with a
predict method that will run the prediction network *without softmax*.  The
model should have variables 

    model.image_size: size of the image (e.g., 28 for MNIST, 32 for CIFAR)
    model.num_channels: 1 for greyscale, 3 for color images
    model.num_labels: total number of valid labels (e.g., 10 for MNIST/CIFAR)

### Running attacks

```python
     from robust_attacks import CarliniL2
     CarliniL2(sess, model).attack(inputs, targets)
```
where inputs are a (*batch x height x width x channels*) tensor and targets are
a (*batch x classes*) tensor. The L2 attack supports a batch_size paramater to
run attacks in parallel. Each attack has many tunable hyper-paramaters. All
are intuitive and strictly increase attack efficacy in one direction and are
more efficient in the other direction.

### Pre-requisites

The following steps should be sufficient to get these attacks up and running on
most Linux-based systems.

```bash
    sudo apt-get install python3-pip
    sudo pip3 install --upgrade pip
    sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```
   
#### To create the MNIST/CIFAR models:

```bash
python3 train_models.py
```

#### To download the inception model:

```bash
python3 setup_inception.py
```

#### And finally to test the attacks

```bash
python3 test_attack.py
```

This code is provided under the BSD 2-Clause, Copyright 2016 to Nicholas Carlini.
