# Neural-Network-Visualiser
## A Neural Network trained on the MNIST database visualised with Pygame.

A Python application that allows you to visualise the training process of a neural network. You can draw digits on the canvas, and the neural network will be trained on the MNIST database to recognise handwritten digits.

## Getting Started

To use the Neural Network Visualiser, follow these steps:

1. Clone the repository to your local machine:

```git clone https://github.com/vSparkyy/Neural-Network-Visualiser.git```

2. Install the required dependencies using `pip`:

```pip install pygame numpy```

3. Run the `main.py` script to start the application:

```python main.py```

## Usage

The application allows you to draw on the canvas using the left mouse button (to draw) and the right mouse button (to erase). Press the 'R' key to reset the canvas. Press the spacebar to start the training process, and you can control the number of training epochs using the epoch slider. Press 'C' to unlock the epoch slider and train the neural network again.

## Neural Network Architecture

The neural network consists of four layers:

1. Input Layer: The input layer has 784 neurons, which corresponds to the 28x28 pixel input images of the MNIST database.

2. Hidden Layer 1: The first hidden layer has 20 neurons.

3. Hidden Layer 2: The second hidden layer also has 20 neurons.

4. Output Layer: The output layer has 10 neurons, each representing the probability of the input digit being one of the ten possible digits (0 to 9).

## Activation Function

The activation function used in the neural network is the sigmoid function. It maps the weighted sum of inputs to a value between 0 and 1, representing the neuron's activation or firing rate.

Sigmoid Function: `1 / (1 + exp(-x))`

## Loss Function

The loss function used for training the neural network is the Binary Cross-Entropy (BCE) loss. It measures the difference between the predicted output and the actual target.

BCE Loss Derivative: `((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)`

## Features

- Real-time visualisation of neural network training
- Interactive canvas for drawing and erasing digits
- Adjustable training epochs
- Percentage bars showing the output probabilities of each digit

## Dependencies

The Neural Network Visualiser requires the following dependencies:

- Python 3.x
- Pygame
- Numpy

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Credits

- Neural Network implementation by vSparkyy
- Assets by Jarmishan

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create an issue or submit a pull request.
