from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
import numpy as np



def main():

    model_file = open("Trained_Models/Auto_Encoder_Trained_Model.json", "r")
    model = model_file.read()
    model_file.close()

    Auto_Encoder = model_from_json(model)
    Auto_Encoder.load_weights("Trained_Models/Auto_Encoder.h5")
    print("Auto_Encoder Model Loaded Successfully")

    #print(Auto_Encoder.summary())

    classifier = Sequential()
    for layer in Auto_Encoder.layers:
        classifier.add(layer)
    for layer in classifier.layers:
        layer.trainable = False

    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(10, activation='softmax'))

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = normalize(xTrain, axis=1)
    xTest = normalize(xTest, axis=1)
    xTrain = np.reshape(xTrain, (len(xTrain), 28, 28, 1))
    xTest = np.reshape(xTest, (len(xTest), 28, 28, 1))

    Test_image = np.reshape(xTest[0], (28, 28))
    plt.imshow(Test_image)
    plt.show()

    print(xTest[0].shape)

    classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy']
                       )
    classifier.fit(xTrain, yTrain,
                   epochs=3,
                   validation_split=.1
                   )


    print(classifier.summary())
    predictions = classifier.predict(xTest)
    print(np.argmax(predictions[0]))

main()