OBJECTIVE:
To implement and evaluate a simple feedforward neural network using TensorFlow v1 on the MNIST dataset, and to analyze the effects of different activation functions and hidden layer sizes through hyperparameter tuning.

Description of the Model
The model is a single hidden-layer feedforward neural network (also known as a multilayer perceptron) designed for image classification on the MNIST dataset. It takes flattened 28x28 grayscale images as input, processes them through one hidden layer using different activation functions (ReLU, Sigmoid, Tanh), and outputs a 10-dimensional vector representing class probabilities for the digits (0–9).


Key points:

Input Layer: 784 neurons (28×28 pixels)
Hidden Layer: Variable size (256, 128, or 64 neurons)
Output Layer: 10 neurons with softmax activation
Activation functions tested: ReLU, Sigmoid, Tanh



Description of Code:


1. Imports and Setup-
Uses tensorflow.compat.v1 to maintain TensorFlow v1 behavior.
Imports standard libraries for data processing (NumPy), visualization (matplotlib, seaborn), and evaluation (sklearn).
2. Data Handling-
Loads MNIST from tensorflow_datasets.
Applies preprocessing: normalization, flattening, and one-hot encoding.
Creates batched datasets for both training and testing.
3. Hyperparameter Tuning-
Explores 3 activation functions and 3 hidden layer sizes (total of 9 combinations).
For each combination:
Defines model architecture and training process.
Trains for 50 epochs with batch size 10.
Tracks loss and accuracy for each epoch.
Evaluates on the test set and stores results.
4. Visualization-
Plots training loss and accuracy for each configuration.
Generates and displays a confusion matrix for classification analysis.
5. Final Output-
Prints the test accuracy and training time for each combination in a summarized format.


Performance Evaluation:

Evaluated using accuracy on the test set and execution time for each configuration.
Results are printed at the end summarizing:
Activation function
Hidden layer size
Final test accuracy
Training and evaluation time
Confusion matrix provides insight into per-class performance and misclassifications.
Loss and accuracy plots show learning behavior across epochs.


MY COMMENTS:


The code provides a clear and practical demonstration of hyperparameter tuning in neural networks.

Impact of Hidden Layer Neuron Count:
256 neurons: Higher capacity to learn complex patterns; may risk overfitting if not regularized.
128 neurons: Balanced choice; performs well while maintaining moderate complexity.
64 neurons: Faster to train, uses less memory but may underfit on complex patterns

Impact of Activation Functions:
ReLU (Rectified Linear Unit): Fast and efficient,Helps avoid vanishing gradient,Performs best in deeper and wide networks.
Sigmoid: Slower convergence,Suffers from vanishing gradient.
Tanh: Better than sigmoid due to output in range [-1, 1],Still slower and prone to saturation compared to ReLU.


Training Time Considerations:
Increasing batch size reduces the number of iterations per epoch → faster training.
Increasing epochs increases the number of full passes over data → longer total training time.

Limitations of the Model:
Uses only one hidden layer — limits representational power.
No regularization — may overfit with larger hidden layers.
No early stopping or learning rate decay — could optimize training further.