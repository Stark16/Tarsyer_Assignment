from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
import numpy as np
import os
import MNIST_Autodecoder

# This script concatenates the classifier layers to the auto encoder layers and then fine
# tunes the entire network trains the classifier layers.

def main():

    # Loading the auto-encoder model:
    model_file = open("Trained_Models/Auto_Encoder_Trained_Model.json", "r")
    model = model_file.read()
    model_file.close()

    Auto_Encoder = model_from_json(model)
    Auto_Encoder.load_weights("Trained_Models/Auto_Encoder.h5")
    print("Auto_Encoder Model Loaded Successfully")

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = normalize(xTrain, axis=1)
    xTest = normalize(xTest, axis=1)
    xTrain = np.reshape(xTrain, (len(xTrain), 28, 28, 1))
    xTest = np.reshape(xTest, (len(xTest), 28, 28, 1))

    # Creating the classifying layers:

    classifier = Sequential()
    # print(Auto_Encoder.summary())

    for layer in Auto_Encoder.layers:
        classifier.add(layer)
    for layer in classifier.layers:
        layer.trainable = False

    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(10, activation='softmax'))

    # Test_image = np.reshape(xTest[0], (28, 28))
    # plt.imshow(Test_image)
    # plt.show()

    print(xTest[0].shape)

    if os.path.exists("./Trained_Models/Classifier_Model.h5") and os.path.exists("./Trained_Models/Classifier_Trained_Model.json") == False:

        classifier.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy']
                               )
        classifier.fit(xTrain, yTrain,
                           epochs=3,
                           validation_split=.1
                           )


        print(classifier.summary())

    print("Model is trained. Now Testing with 10 images:")

    noise_factor = 0.5
    xTest_noisy = []
    xTest_noisy = xTest + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=xTest.shape)
    xTest_noisy = np.clip(xTest_noisy, 0., 1.)

    prediction = classifier.predict(xTest_noisy)

    print(np.argmax(prediction[0]))

    plt.subplot(2, 1, 1)
    plt.imshow(xTest_noisy[0].reshape(28, 28))
    plt.subplot(2, 1, 2)
    plt.imshow(xTest[0].reshape(28, 28))
    plt.show()





main()