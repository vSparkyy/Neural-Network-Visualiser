import numpy as np
import pathlib


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    labels = np.eye(10)[labels]
    for index, digit in enumerate(images):
        images[index] = np.where(digit >= 0.5, 1, 0)
    return images, labels
