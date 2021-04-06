import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

SAMPLE_PATH = os.getcwd() + '/chest_Xray/'
NORMAL_VALUE = np.uint8(1)
PNEUMONIA_VALUE = np.uint8(0)

TEST_NORMAL_PATH = SAMPLE_PATH + "test/NORMAL/"
TEST_PNEUMONIA_PATH = SAMPLE_PATH + "test/PNEUMONIA/"

TRAIN_NORMAL_PATH = SAMPLE_PATH + 'train/NORMAL/'
TRAIN_PNEUMONIA_PATH = SAMPLE_PATH + 'train/PNEUMONIA/'

VAL_NORMAL_PATH = SAMPLE_PATH + 'val/NORMAL/'
VAL_PNEUMONIA_PATH = SAMPLE_PATH + 'val/PNEUMONIA/'


class Data:
    def __init__(self, which="all"):
        self.img_height = 172
        self.img_width = 364
        if which == "all":
            self.import_train()
            self.import_test()
            self.import_val()
        else:
            if which == "train":
                self.import_train()
            elif which == "test":
                self.import_test()
            else:
                self.import_val()

    def empty(self):
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def import_train(self):
        list_train_images = []
        y_train = []

        for image in os.listdir(TRAIN_NORMAL_PATH):
            if image.split('.')[len(image.split('.')) - 1] == "jpeg":
                img = load_img(TRAIN_NORMAL_PATH + image, color_mode="grayscale",
                               target_size=(self.img_height, self.img_width))
                img = img_to_array(img).astype('float32')
                list_train_images.append(img)
                y_train.append(NORMAL_VALUE)

        img = None

        for image in os.listdir(TRAIN_PNEUMONIA_PATH):
            if image.split('.')[len(image.split('.')) - 1] == "jpeg":
                img = load_img(TRAIN_PNEUMONIA_PATH + image, color_mode="grayscale",
                               target_size=(self.img_height, self.img_width))
                img = img_to_array(img).astype('float32')
                list_train_images.append(img)
                y_train.append(PNEUMONIA_VALUE)

        img = None
        x_train = np.array(list_train_images, dtype=np.float32)
        list_train_images = None
        y_train = np.array(y_train, dtype=np.float32)

        x_train /= 255

        self.x_train = x_train
        self.y_train = y_train
        x_train = None
        y_train = None

        print('Train Shape: ' + str(self.x_train.shape))

    def import_test(self):
        list_test_images = []
        y_test = []

        for image in os.listdir(TEST_NORMAL_PATH):
            if image.split('.')[len(image.split('.')) - 1] == "jpeg":
                img = load_img(TEST_NORMAL_PATH + image, color_mode="grayscale",
                               target_size=(self.img_height, self.img_width))
                img = img_to_array(img).astype('float32')
                list_test_images.append(img)
                y_test.append(NORMAL_VALUE)

        for image in os.listdir(TEST_PNEUMONIA_PATH):
            if image.split('.')[len(image.split('.')) - 1] == "jpeg":
                img = load_img(TEST_PNEUMONIA_PATH + image, color_mode='grayscale',
                               target_size=(self.img_height, self.img_width))
                img = img_to_array(img).astype('float32')
                list_test_images.append(img)
                y_test.append(PNEUMONIA_VALUE)

        img = None
        x_test = np.array(list_test_images, dtype=np.float32)
        list_test_images = None
        y_test = np.array(y_test, dtype=np.float32)

        x_test /= 255

        self.x_test = x_test
        self.y_test = y_test
        x_test = None
        y_test = None

        print('Test Shape: ' + str(self.x_test.shape))

    def import_val(self):
        list_val_images = []
        y_val = []

        for image in os.listdir(VAL_NORMAL_PATH):
            if image.split('.')[len(image.split('.')) - 1] == "jpeg":
                img = load_img(VAL_NORMAL_PATH + image, color_mode="grayscale",
                               target_size=(self.img_height, self.img_width))
                img = img_to_array(img).astype('float32')
                list_val_images.append(img)
                y_val.append(NORMAL_VALUE)

        img = None

        for image in os.listdir(VAL_PNEUMONIA_PATH):
            if image.split('.')[len(image.split('.')) - 1] == "jpeg":
                img = load_img(VAL_PNEUMONIA_PATH + image, color_mode="grayscale",
                               target_size=(self.img_height, self.img_width))
                img = img_to_array(img).astype('float32')
                list_val_images.append(img)
                y_val.append(PNEUMONIA_VALUE)

        img = None
        x_val = np.array(list_val_images, dtype=np.float32)
        list_val_images = None
        y_val = np.array(y_val, dtype=np.float32)

        x_val /= 255

        self.x_val = x_val
        self.y_val = y_val
        x_val = None
        y_val = None

        print('Val Shape: ' + str(self.x_val.shape))
