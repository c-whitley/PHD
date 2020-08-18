import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import Conv2D


class UNet:

    def __init__(self, **kwargs):

        self.n_classes = kwargs.get(n_classes)

    def make_unet(self):

    	layers = [
    	tf.keras.layers(Conv2D(32, (3,3),)),
    	]

        self.model = Sequential()
        self.model.add(Conv2D(32, ))

print("Hello")
