OBJECTIVE:

This project implements a Convolutional Neural Network (CNN) for classifying images from the Fashion MNIST dataset. The main goal is to experiment with different hyperparameters (filter size, regularization strength, batch size, and optimizer) to analyze their impact on model performance.

Description of the Model:

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. It follows a standard CNN architecture designed for image classification:

1. Convolutional Layers:
The model starts with a Conv2D layer with 32 filters and a kernel size of either 3x3 or 5x5 (depending on the hyperparameter setting).
A second Conv2D layer with 64 filters is added.
Both convolutional layers use ReLU activation and L2 regularization (with values 0.0001 or 0.001) to prevent overfitting.

2. Pooling Layers:
Each convolutional layer is followed by a MaxPooling2D layer (2x2) to reduce spatial dimensions and computational load.

3. Fully Connected Layers:
After flattening the feature maps, a Dense layer with 128 neurons and ReLU activation is added to learn high-level features.
The final output layer has 10 neurons with softmax activation, corresponding to the 10 classes in Fashion MNIST.

4. Optimizer & Loss Function:
The model uses either Adam or SGD optimizer (depending on the experiment).
The loss function is sparse_categorical_crossentropy since this is a multi-class classification problem.



Description of Code:

1. Data Loading & Preprocessing:
The Fashion MNIST dataset is loaded using keras.datasets.fashion_mnist.
The pixel values of images are normalized (divided by 255.0) to bring them to the range [0,1].
The dataset is reshaped to add a channel dimension (from (28,28) to (28,28,1)) for compatibility with the CNN.


2. Defining the CNN Model:
A function create_model is implemented, allowing customization of hyperparameters:
filter_size: Defines kernel size (3x3 or 5x5).
regularization: Applies L2 regularization to prevent overfitting.
optimizer: Defines the optimization algorithm (Adam or SGD).
The model is compiled with sparse_categorical_crossentropy loss and accuracy as the evaluation metric.

3. Hyperparameter Tuning & Training:
Four hyperparameters are tested in various combinations:
Filter sizes: [3, 5]
Regularization values: [0.0001, 0.001]
Batch sizes: [32, 64]
Optimizers: ['adam', 'sgd']

A nested loop iterates over all possible configurations, training a model for each.
Each model is trained for 5 epochs with validation data included.
The test accuracy for each model is recorded and printed.

4. Performance Visualization:
The validation accuracy for each hyperparameter combination is plotted using Matplotlib.
This helps visualize how different configurations affect the model’s performance.



Performance Evaluation:

Each model's performance is evaluated based on validation accuracy.
Test accuracy is recorded for all experiments.
A plot is generated to compare how hyperparameter choices affect validation accuracy across epochs.



MY COMMENTS:

Filter Size (3x3 vs 5x5):
A 3x3 filter is smaller and faster. It works well for most tasks and helps avoid overfitting.
A 5x5 filter looks at a bigger area of the image, which might help learn more details, but it’s slower and can sometimes overfit the data.

Regularization (None vs L2 = 0.001):

L2 regularization helps prevent overfitting by keeping the model’s weights small.
Using no regularization lets the model learn freely, but it might memorize the training data instead of learning patterns.

Batch Size (32 vs 64):

Batch size 32 trains slower but often gives better results in the long run.
Batch size 64 trains faster but may not perform as well on new, unseen data.

Optimizer (Adam vs SGD):

Adam is fast and easy to use—it usually gives good results without much tuning.
SGD is slower and needs more tweaking (like setting the right learning rate), but it can sometimes give better results.

Limitations of This Experiment:
Short Training Time:
The models are trained for only 10 epochs, which might not be enough. Training longer could improve accuracy.

Limited Testing of Options:
Only a few values for each setting (like filter size and batch size) are tested. Trying more combinations could give better insights.

Simple Dataset:
This experiment uses Fashion MNIST, which has small, black-and-white images. The results might be very different on colorful or more complex datasets like CIFAR-10.

GPU Use:
The code checks for a GPU to speed things up, but how fast it runs depends on whether a GPU is actually available.

