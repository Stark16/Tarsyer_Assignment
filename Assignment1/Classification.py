from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
import numpy as np
import os
from Assignment1 import MNIST_Autodecoder

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

    xTrain = xTrain.astype('float32') / 255.
    xTest = xTest.astype('float32') / 255.
    xTrain = np.reshape(xTrain, (len(xTrain), 28, 28, 1))
    xTest = np.reshape(xTest, (len(xTest), 28, 28, 1))

    noiseFactor = 0.5
    xTrain_noisy = xTrain + noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=xTrain.shape)
    xTest_noisy = xTest + noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=xTest.shape)

    xTrain_noisy = np.clip(xTrain_noisy, 0., 1.)
    xTest_noisy = np.clip(xTest_noisy, 0., 1.)

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

    # Now I check if classifier model is trained or not, if not the script trains it or else ask the user to decide if they
    # wish to retrain the model.

    ch = int(input("do you wish to retrain the Classification model? 1 for yes, 2 for no."))

    if os.path.exists("./Trained_Models/Classifier_Model.h5") and os.path.exists("./Trained_Models/Classifier_Trained_Model.json") == False or ch == 1:

        classifier.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy']
                               )
        history = classifier.fit(xTrain_noisy, yTrain,
                           epochs=3,
                           validation_split=.1
                           )
        Classifier_model_trained = classifier.to_json()

        with open("Trained_Models/Classifier_Trained_Model.json", 'w') as json_model:
            json_model.write(Classifier_model_trained)
        classifier.save_weights("Trained_Models/Classifier.h5")
        json_model.close()
        print("\n\t Classifier Model has been trained and Saved Successfully! ")
        print("\tYou can Find Trained Models Under Trained_Model Directory")

        # Plotting curves:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        print(classifier.summary())

    print("Model is trained. Now Testing with 10 images:")

    # I test the model by adding noise to the input image and then passing them through the entire network
    # to see if it can classify them

    noise_factor = 0.5
    xTest_noisy = []
    xTest_noisy = xTest + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=xTest.shape)
    xTest_noisy = np.clip(xTest_noisy, 0., 1.)

    prediction = classifier.predict(xTest_noisy)

    for i in range(10):
        print(np.argmax(prediction[i]))

        plt.subplot(2, 1, 1)
        plt.imshow(xTest_noisy[i].reshape(28, 28))
        plt.subplot(2, 1, 2)
        plt.imshow(xTest[i].reshape(28, 28))
        plt.show()





main()