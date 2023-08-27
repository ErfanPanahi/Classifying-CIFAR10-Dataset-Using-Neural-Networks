![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/23875f1f-d42c-4181-bed0-3c096cbec28e)# Classifying-CIFAR10-Dataset-Using-Neural-Networks
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

*Problem 1.* Choosing the Most Suitable Batch Size

In this question, we consider the other parameters as follows:
Batch-Size: {32, 64, 256}
Activation Functions: {Layer #1: 'ReLU', Layer #2: 'ReLU'}
Optimizer: SGD (learning-rate = 0.01, momentum = 0.9)
Loss Function: Categorical Cross Entropy
(We consider 20 epochs in this section.)
The images below depict the desired outputs for this question. As observed, for a batch length of 256, the machine's accuracy and learning speed are improved.

*batch-size = 32:*

![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/c145d394-f888-4c28-a8f7-71ed872a6d6c)

![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/80df75af-dd47-4f29-8505-94b610416b70)

![image](https://github.com/ErfanPanahi/Classifying-CIFAR10-Dataset-Using-Neural-Networks/assets/107314081/4cd3fc1e-732d-4e22-93f8-20400f43497c)

