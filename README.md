[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/MYTd_5jm)
# 5AUA0 Group 7

Student name - Student number - Email address

Mingdao Lin - 1864416 - m.lin@student.tue.nl

Wei Zheng - 1886061 - w.zheng@student.tue.nl


## Requirements
This code has been tested with the following versions:
- python == 3.10
- pytorch == 2.0
- torchvision == 0.15
- numpy == 1.23
- pillow == 9.4
- cudatoolkit == 11.7 (Only required for using GPU & CUDA)

We recommend you to install these dependencies using [Anaconda](https://docs.anaconda.com/anaconda/install/). With Anaconda installed, the dependencies can be installed with
```bash
conda create --name 5aua0 python=3.10
conda activate 5aua0
conda init
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```     

## Dataset
To download and prepare the dataset, follow these steps:
- Download the [SemKITTI-DVPS dataset]([https://www.cs.toronto.edu/~kriz/cifar.html](https://github.com/HarborYuan/PolyphonicFormer)) from [this URL](https://huggingface.co/HarborYuan/PolyphonicFormer/resolve/main/semkitti-dvps.zip).
- Unpack the file and store the `cifar-10-batches-py` directory in the `DATALOC` directory of this repository.
- It should look like this:

DATALOC

|── semkitti-dvps

│   ├── video_sequence

│   │   ├── train

│   │   │   ├── 000000_000000_leftImg8bit.png

│   │   │   ├── 000000_000000_gtFine_class.png

│   │   │   ├── 000000_000000_gtFine_instance.png

│   │   │   ├── 000000_000000_depth_718.8560180664062.png

│   │   │   ├── ...

│   │   ├── val

│   │   │   ├── ...

## Training and testing the network
To train the network of one frame in folder named VIP_deeplab_one_frame, run:
```bash
python train0622.py 
```
To train the network of two frames in folder named VIP_deeplab_two_frames, run:
```bash
python train_CE_0622.py
```
or
```bash
python train_deep07.py
```

## Config and hyperparameters
We define:
- `batch_size_train`: The batch size during training (default: 8)
- `lr`: The learning rate (default: 1e-3)
- `lr-scheduler`: T_0 = 5, T_mult=2
- `num_iterations`: The number of iterations to train the network (default: 60)
- `checkpoint`: The number of model saved as checkpoints (default: 5)
- `DeepLabCE`: The top-k pixel are focused on (default: 0.7)


## Potential improvements
As mentioned, this is a very minimal example. This code should be changed to solve your task, and there are plenty of functionalities that you could add to make your life easier, or to improve the performance. Some examples:
- Store the `config` info for each training that you do, so you remember the hyperparameters you used.
- Run evaluation on the validation/test set after every X steps, to keep track of the performance of the network on unseen data.
- Try out other network components, such as `BatchNorm` and `Dropout`.
- Fix random seeds, to make results (and bugs) better reproducible.
- Visualize the results of your network! By visualizing the input image and printing/logging the predicted result, you can get insight in the performance of your model.
- And of course, improve network and the hyperparameters to get a better performance!

## HPC
In order to run this repository on the Snellius High Performance Computing platform, please follow our [HPC Tutorial](https://tue-5aua0.github.io/hpc_tutorial.html)
