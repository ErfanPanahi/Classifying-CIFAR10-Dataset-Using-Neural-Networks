# Classifying-CIFAR10-Dataset-Using-Neural-Networks
In this repository, I am working on classifying the CIFAR10 dataset using neural networks. To achieve this, I initially perform classification using an MLP network. Then, to enhance the network's performance and reduce learning time, I employ convolutional layers, pooling, batch normalization, and dropout layers.

First, we enter the data using the provided instructions in the following format.

    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

Then, the training data is divided into two sections: train and validation. We pay attention to the fact that, considering batch sizes of 32, 64, or 256, the training data should be a multiple of these three sizes. For this purpose, out of 50,000 training data points, approximately 90%, equivalent to 45,056 data points, are allocated to the train set, and the rest are assigned to validation.

*Of course, we could choose not to do this and allocate exactly 90% of the data to the train set, but for two reasons, we attempt the allocation as described above:*

1- Training is performed on all data points.
2- If the number of training data points is not a multiple of batch sizes, some data points are not used for training, which might affect the accuracy of the model.

Additionally, we use the `to_categorical()` function from the Keras library to convert labels into one-hot encoded format.

Finally, as shown in the image below, we display the first 10 images of the train dataset.

![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/0a16ba42-25d9-4ef6-abd7-1ceecad21752)

Now we want to design a network using the Keras library to classify data. In the first part, we design a network with two hidden layers, and in the second part, we add convolutional layers to the designed network to enhance its performance.

**Part A:** Using the MLP Network

In this section, we first design a network with two hidden layers. Through trial and error, we conclude that setting the lengths of the hidden layers to 200 and 100 yields favorable results (the parameters influencing the choice of these lengths are accuracy and learning speed). To implement the network, a function called MLP() has been written, to which we provide the variable parameters of the problem as inputs.

***Problem 1.*** Choosing the Most Suitable Batch Size

In this question, we consider the other parameters as follows:

- Batch-Size: {32, 64, 256}
- Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}
- Optimizer: SGD (learning-rate = 0.01, momentum = 0.9)
- Loss Function: Categorical Cross Entropy

(We consider 20 epochs in this section.)

The images below depict the desired outputs for this question. As observed, for a batch length of 256, the machine's accuracy and learning speed are improved.

***batch-size = 32 :***

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/c145d394-f888-4c28-a8f7-71ed872a6d6c />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/80df75af-dd47-4f29-8505-94b610416b70 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/4cd3fc1e-732d-4e22-93f8-20400f43497c />
</p>

***batch-size = 64 :***

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/f42fb11e-d823-40f8-b49b-4ff9ef7eee20 />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/a6d5896b-3235-405d-9c85-10010642f4e8 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/efa4df17-e18f-4e9e-956a-aea25356cedf />
</p>

***batch-size = 256 :***

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/fb896bf5-53a2-4f50-937d-902ee5b61429 />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/0bbd8e09-7bee-4aba-bcb6-0a88778f00ab />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/f9ef2b4f-4e45-41bf-bef5-2baae9a1c8f4 />
</p>

***Problem 2.*** Choosing the most suitable activation functions for the hidden layers
In this question, we consider the following parameters:
- Batch-Size = 256
- Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}
- {Layer #1: 'tanh', Layer #2: 'tanh'}
- {Layer #1: 'ReLU', Layer #2: 'sigmoid'}
- Optimizer: SGD (learning-rate = 0.01, momentum = 0.9)
- Loss Function: Categorical Cross Entropy
Figures below depict the desired output results for this question. Additionally, the first configuration (ReLU, ReLU) was examined in question 1. As observed, the performance of machine learning is better for the first configuration (ReLU, ReLU).

***activation functions = {tanh, tanh}***

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/7bc6303a-05bd-4169-8f3a-7d84629f250a />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/c12228b3-a189-4da0-a20e-1425749847db />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/c8a9e0f7-65d9-402c-9404-253aa1a16468 />
</p>


***activation functions = {RelU, sigmoid}***

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/6db97e29-cdab-47a4-adea-fc774cc23025 />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/b86f20c4-bc22-451a-b103-9e0f001098c4 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/0f15e71c-c155-4f16-b007-a16bf7b98e27 />
</p>

***Problem 3.*** Choosing the Most Appropriate Error Function

In this question, we consider the following parameters:

- Batch Size: 256
- Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}
- Optimizer: SGD (Learning Rate = 0.01, Momentum = 0.9)
- Loss Function: {Categorical Cross-Entropy, Poisson}

The images below illustrate the desired outputs for this question. Furthermore, the first function, Categorical Cross-Entropy, was examined in question 1. As observed, the machine learning performance is better for the Categorical Cross-Entropy error function.

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/f685ce99-efca-43ef-8fa0-5c91b7a86e4f />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/17cb0421-a16a-4354-89e4-9ccf10a65944 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/3c1f54d5-c0ea-4d18-90e3-916eb60c565c />
</p>

***Problem 4.*** Choosing the most suitable optimizer

In this question, we consider the following parameters:

Batch-Size = 256
Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}
Optimizer: SGD (learning-rate = 0.01, momentum = 0.9)
Adam (learning-rate = 0.01)
Loss Function: {Categorical Cross Entropy, Poisson}
Figures 2-5 depict the desired outputs for this question. Furthermore, the first optimizer, SGD, was examined in question 1. As observed, the machine learning performance is better for the SGD optimizer.
