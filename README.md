![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/d67226ff-918f-4100-95e6-c0ee44a71190)# Classifying-CIFAR10-Dataset-Using-Neural-Networks
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
- Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}, {Layer #1: 'tanh', Layer #2: 'tanh'}, {Layer #1: 'ReLU', Layer #2: 'sigmoid'}
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

- Batch-Size = 256
- Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}
- Optimizer: SGD (learning-rate = 0.01, momentum = 0.9) , Adam (learning-rate = 0.01)
- Loss Function: {Categorical Cross Entropy, Poisson}

Figures below depict the desired outputs for this question. Furthermore, the first optimizer, SGD, was examined in question 1. As observed, the machine learning performance is better for the SGD optimizer.

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/6778121c-87f6-4476-9533-6788ab84c122 />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/78b65157-a469-43dd-a27b-f682dc4802e3 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/cb3db281-c5d4-4489-a9c2-d4af67d7eb0f />
</p>

Selecting the most suitable parameters for the network
Based on the obtained results, the optimal parameters that lead to better accuracy and speed in the network's learning performance will be as follows:
- Batch-Size = 256
- Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}
- Optimizer: SGD (learning-rate = 0.01, momentum = 0.9)
- Loss Function: Categorical Cross Entropy, Poisson

The following image summarizes the network's layers with the aforementioned parameters.

![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/e88d873d-baff-4b18-8033-5b703a802e7e)

**Part 2:** Using MLP+CNN Network

***Problem 1.*** The Impact of Adding Convolutional Layers

Now, we want to add two convolutional layers to the best-designed convolutional network from the previous section, as shown in question 5, part A. We will examine the effect of adding these layers on the network's accuracy. Considering that the network becomes very slow in this section, we set the number of epochs to 10. (The function written for this part is denoted as CNN().)

The image below depicts the error and accuracy graph along with the compared parameter values and a summary of the network's layers. As observed, the accuracy has significantly increased, and the error has decreased. However, the noteworthy point here is that the network operates very slowly. In the following questions, we aim to implement techniques that maintain the network's accuracy despite the increased speed.

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/4af96a78-5a4d-4389-abc5-3b03e84cfb60 />
</p>

<p align="center">
  <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/892b8a3f-5f67-4f49-9a6d-418cc71d9ff0 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/24c50708-b361-42bc-8ac3-202944c39e11 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/b2f0e7f9-e274-47cc-99f2-2e6072ee9c03 />
</p>

***Problem 2.*** The Impact of Adding Pooling and Batch Normalization Layers

In this question, we first provide an explanation about these two layers and then examine their impact on the network.

Pooling Layer: In neural networks, after adding convolutional layers and as the dimensions and sizes increase in the input to the hidden layers, a Pooling layer is employed to reduce the size and shape of the output from the convolutional layer. Figure below illustrates the functioning of this layer. Incorporating this layer, despite potentially not significantly affecting accuracy, greatly reduces complexity and accelerates the network's speed.

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/d3e94df8-0bc3-4970-b12d-9fec89cbed45 />
</p>

*Batch Normalization Layer:* The Batch Normalization layer is used in neural networks to accelerate the learning process. It allows us to use a higher learning rate during optimization. This layer employs a normalization technique where, instead of normalizing the entire dataset, we normalize the data within each mini-batch.

The following image illustrates the error and accuracy graphs along with the compared parameter values, along with a summary of the network layers after adding the Pooling and Batch Normalization layers. As observed, the learning speed of the network has significantly increased. Furthermore, both accuracy and error metrics have not only remained stable but have also shown improvement. A notable point in this question, as well as the previous one, is that after a certain epoch, the model starts to experience overfitting, leading to a decrease in its accuracy.

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/451e6cfa-7860-4fb3-b054-c13e9f1c9102 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/7f4ea640-58fd-4ecd-8c91-e914e929d3a9 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/b35d1462-7948-4e98-b106-0f4b5459b4b4 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/1a817e75-672f-49c1-8c37-10e2ebea734e />
</p>

***Problem 3.*** The Impact of Adding Dropout Layers

Dropout Layer: In neural networks, some neurons may have a negative impact on the continuation of the network, and it's necessary to deactivate them in subsequent layers during the Feed-Forward process. To achieve this, the dropout layer is utilized. In this layer, a percentage of output neurons from the previous layer are made ineffective in the following layer. This action will not only increase the network's speed but also have an effect on its accuracy. The image below illustrates an example of how this layer functions. One of the most important characteristics of this layer is its ability to prevent Overfitting.

![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/1d2aadbc-b38a-4e28-974c-f5168f7920fa)

The following image depicts the error and accuracy graph along with the compared parameters, along with a summary of the network layers after the addition of Dropout layers. As observed, the network's learning speed has significantly increased, and the accuracy has not decreased. Additionally, during network testing iterations, it can be seen that its accuracy remains stable, making it more resistant to overfitting.

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/c94ac696-f0f8-402c-a59c-080679e436ba />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/1120c9ba-679d-4cf2-860b-16cde373bf53 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/a96ebed1-3637-4d7e-b71b-79c19761ce41 />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/635faba5-c081-48b9-ae4b-a19db68024bf />
</p>

***Problem 4.*** Early Stop

In neural networks, after a few epochs, it is observed that the validation (or test) data error curve deviates from the training data error curve and starts to increase inversely. This indicates that the network's performance on the test data is deteriorating, and therefore, it is necessary to stop the network's learning process. The solution to this problem is Early Stop. The following illustration indicates the point at which we need to apply early stopping in the network's learning process.

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/8eaa01d8-4f91-44d0-a430-21c739ddf52f />
</p>

*Criteria used in early stopping:* We employ two criteria to detect when to halt learning in a network. One involves monitoring the gap between the accuracy graphs of evaluation data and training data, while the other entails observing the gap between the error graphs of these two data sets. Furthermore, it is important to determine the extent of sensitivity to the gaps in these two graphs. Using the `EarlyStopping()` function within the keras library, these criteria can be established.

The image below illustrates the output parameters following the implementation of early stopping on the neural network. As observed, at epoch 19, after considering the gap in the validation data error graph (determined by the `patience` parameter), this ascending graph ceases to rise, thus concluding the learning process. Overall, even if a greater number of epochs were considered, should the increment continue, the network's learning would halt.

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/c2827361-e8f8-4a92-80d1-20210703624e />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/56bea9b0-4655-4724-876e-99220222913e />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/8c99393f-72c0-4620-b461-69b9ca9ee4fe />
</p>

<p align="center">
      <img src=https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/c23370c6-6706-43e2-8280-f81c6e8153fa />
</p>



