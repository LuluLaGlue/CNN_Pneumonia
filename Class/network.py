from tensorflow import nn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D, SeparableConv2D, BatchNormalization
import matplotlib.pyplot as plt


class Network:
    def __init__(self, shape):
        # Using the Sequential model as it allows linear stacking of layers

        self.model = Sequential()
        self.createNet()

    def createNet(self):
        # Conv2D are convolutional layers that scans img in order to detect carasteristical fragments.
        # it outputs a {first param} tab in 2D.
        # padding='same' is used to determine how the 'scanner' goes through the image.
        # activation='relu' function applied to all 'scanned' element -> g(z) = max(0, z)

        self.model.add(Conv2D(16, 3, padding='same', activation='relu'))

        # After Conv we use MaxPooling to go through the 2D tab and simplify it by taking max value of near pxs.
        # Could also use AveragePooling2D to get the average

        self.model.add(MaxPooling2D())
        # self.model.add(AveragePooling2D())

        self.model.add(Conv2D(32, 3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D())
        # self.model.add(AveragePooling2D())

        self.model.add(Conv2D(64, 3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D())
        # self.model.add(AveragePooling2D())

        # Dropout(0.2) randomly deletes 20% of the output units

        self.model.add(Dropout(0.2))

        # Flatten() allows to flatten the output of the previous layers to have a 1D object

        self.model.add(Flatten())

        # Dense() are 2 fully connected layers allowing to connect the first output of Conv and MAxPooling to our output layer of 2 neurons
        # THe last layer is composed of 2 neurons as we have only 2 possible outputs (Sick or OK)
        # softmax takes in a score and outputs a probability which allows to know how good our model is.

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation=nn.sigmoid))

    def empty(self):
        self.model = None

    def draw(self, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        # plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training Loss')
        plt.show()
