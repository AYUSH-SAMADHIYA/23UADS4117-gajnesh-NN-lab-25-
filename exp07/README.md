## Objective:
The objective of this project is to develop a deep learning model capable of classifying medical images into different categories using a transfer learning approach with a pretrained ResNet-18 model.

## Description of Model:
This project utilizes ResNet18, a convolutional neural network architecture pretrained on the ImageNet dataset. The model is fine-tuned by replacing its final fully connected layer to match the number of classes in the provided medical dataset.

Backbone: ResNet18 (pretrained)

Modification: Output layer adjusted for custom class count

Loss Function: Cross-Entropy Loss

Optimizer: Adam (learning rate = 0.001)

## Description of Code:
Libraries Used: PyTorch, Torchvision, Matplotlib

### Dataset Handling:
Images are loaded from a directory structure with train and val subfolders.
Data augmentation (e.g., random horizontal flip) and normalization are applied.

### Model Definition:
A pretrained ResNet-18 model is loaded.
The final FC layer is modified to output predictions based on the number of classes in the dataset.

### Training Loop:
Training is done for 10 epochs.
Tracks training and validation loss and accuracy.
Best-performing model on the validation set is saved.

### Visualization:
A loss curve is plotted at the end of training to visualize training progress.



## Performance Evaluation:
### Training & Validation Metrics:

Accuracy and loss are printed for both training and validation phases across epochs.
The best model is selected based on highest validation accuracy.

### Visualization:

A loss curve is generated to track the decrease in loss over training epochs
Actual performance metrics (accuracy, loss) may vary depending on the dataset quality and number of samples.


## My Comments:

The modular structure of the training function makes it easy to extend and experiment with different models or training strategies.

For further improvements, you might consider:

Adding test set evaluation
Experimenting with deeper networks (e.g., ResNet-50)
Implementing learning rate scheduling or early stopping

Model Limitations:
If the dataset has fewer images or uneven classes, the model might get confused.
ResNet-18 is not super deep. It works, but better models exist.
Didn’t handle bad or wrong images in the dataset.
