"""modelDefinition.py"""

# Package imports
import chainer.functions as F
from chainer import serializers
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def normalize(x):
    x_len = np.linalg.norm(x, keepdims=True, axis=1)
    x = x / x_len
    return x



# Create an instance of the network you trained
def save_model(model):
    #model = MLP()
    serializers.save_npz('MyMnistWMem.model', model)


# Get a test image and label
def test_images(model, test):
    # Load the saved paremeters into the instance
    serializers.load_npz('MyMnistWMem.model', model)
    count = 100
    for i in range(count):
        x, t = test[i]

        # forward calculation of the model by sending X
        x = x[None, ...]
        test_one_image(model, x, t)


def test_one_image(model, x, t):
    y = model.predictor(x)

    # The result is given as Variable, then we can take a look at the contents by the attribute, .data.
    y = y.data
    # Look up the most probable digit number using argmax
    pred_label = y.argmax(axis=1)
    print('predicted label:', pred_label[0])
    print('label:', t)
    if pred_label[0] != t:
        plt.savefig('7.png')
        plt.imshow(x.reshape(28, 28), cmap='gray')
        pass


