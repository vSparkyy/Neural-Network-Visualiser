import numpy as np
import data


class Layer:
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize a layer of the neural network.

        Args:
            input_size (int): The number of input neurons.
            output_size (int): The number of output neurons.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-0.5, 0.5, (self.output_size, self.input_size))
        self.bias = np.zeros((output_size, 1))
        self.output = None

    def forward_propagate(self, inputs: np.ndarray) -> None:
        """
        Perform forward propagation on the layer.

        Args:
            inputs (np.ndarray): The input data to the layer.
        """
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(self.weights, self.inputs) + self.bias)

    def backward_propagate(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform backward propagation on the layer.

        Args:
            output_gradient (np.ndarray): The gradient of the loss function with respect to the layer's output.
            learning_rate (float): The learning rate for updating the weights and biases.

        Returns:
            np.ndarray: The gradient of the loss function with respect to the layer's input.
        """
        delta = output_gradient * self.sigmoid_derivative(self.output)
        self.weights += -learning_rate * np.dot(delta, np.transpose(self.inputs))
        self.bias += -learning_rate * delta

        return np.dot(np.transpose(self.weights), delta)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output data after applying the sigmoid function element-wise.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: The derivative of the sigmoid function.
        """
        return x * (1 - x)


class NeuralNetwork:
    def __init__(self):
        self.images, self.labels = data.get_mnist()
        self.img = None
        self.accuracy = None
        self.current_epoch = 0
        self.learning_rate = 0.1
        self.correct = 0
        self.layers = [
            Layer(784, 20),
            Layer(20, 20),
            Layer(20, 10),
        ]

    def forward_propagate(self, image: np.ndarray) -> None:
        """
        Perform forward propagation on the neural network.

        Args:
            image (np.ndarray): The input image to propagate through the network.
        """
        self.image = image
        for layer in self.layers:
            layer.forward_propagate(self.image)
            self.image = layer.output

    def backward_propagate(self, label: np.ndarray) -> None:
        """
        Perform backward propagation on the neural network.

        Args:
            label (np.ndarray): The target label corresponding to the input image.
        """
        error = self.bce_derivative(label, self.layers[-1].output)
        for layer in reversed(self.layers):
            error = layer.backward_propagate(error, self.learning_rate)

    def bce_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the binary cross-entropy loss function.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted probabilities.

        Returns:
            np.ndarray: The gradient of the loss function.
        """
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    def train(self, epochs: int = 10) -> None:
        """
        Train the neural network for the specified number of epochs.

        Args:
            epochs (int, optional): The number of epochs to train. Defaults to 10.
        """
        self.epochs = epochs
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            for img, lbl in zip(self.images, self.labels):
                self.img = img.reshape(-1, 1)
                self.lbl = lbl.reshape(-1, 1)
                self.forward_propagate(self.img)
                self.correct += int(
                    np.argmax(self.layers[-1].output) == np.argmax(self.lbl))
                self.backward_propagate(self.lbl)
            self.accuracy = str(
                round((self.correct / self.images.shape[0]) * 100, 2))
            self.correct = 0

    def get_percentages(self, layer: np.ndarray) -> list:
        """
        Get the percentages of the output neurons.

        Args:
            layer (np.ndarray): The output of the last layer.

        Returns:
            list: A list of tuples containing the neuron index and percentage value.
        """
        percentages = []
        if layer is not None:
            for index, weight in enumerate(np.round((layer / np.sum(layer)), 3)):
                percentages.append((index, f"{weight[0]:.3f}"))
            return percentages

    def test(self, inputs: np.ndarray) -> None:
        """
        Test the neural network on the given inputs.

        Args:
            inputs (np.ndarray): The input data to test the network.
        """
        self.forward_propagate(inputs.reshape(-1, 1))
