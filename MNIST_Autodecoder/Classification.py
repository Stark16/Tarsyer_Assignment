from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt


def main():

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTrain = normalize(xTrain, axis=1)
    xTest = normalize(xTest, axis=1)

    Classifier = Sequential()
    Classifier.add(Flatten())
    Classifier.add(Dense(128, activation='relu'))
    Classifier.add(Dense(128, activation='relu'))
    Classifier.add(Dense(10, activation='softmax'))

    model_file = open("Trained_Models/Auto_Encoder_Trained_Model.json", "r")
    model = model_file.read()
    model_file.close()

    Auto_Encoder = model_from_json(model)
    Auto_Encoder.load_weights("Trained_Models/Auto_Encoder.h5")
    print("Auto_Encoder Model Loaded Successfully")

    Classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy']
                       )
    history = Classifier.fit(xTrain, yTrain,
                   epochs=3,
                   validation_data=(xTest, yTest)
                             )

    Classifier_Trained = Classifier.to_json()
    with open("Trained_Models/Classifier_Trained_Model.json", "w") as json_model:
        json_model.write(Classifier_Trained)
    Classifier.save_weights("Trained_Models/Classifier_Model.h5")
    json_model.close()

    print("\n\t Classifier Model has been trained and Saved Successfully!")
    print("\t You can Find Trained Models Under Trained_Model Directory")

    plt.plot(history.history['accuracy'])
    plt.title('Classifier model accuracy')
    plt.ylabel('accurace')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.plot()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Classifier model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

main()