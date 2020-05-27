import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from keras.datasets import mnist


def main():

    (xTrain, _), (xTest, _) = mnist.load_data()

    xTrain = xTrain.astype('float32') / 255.
    xTest = xTest.astype('float32') / 255.
    xTrain = np.reshape(xTrain, (len(xTrain), 28, 28, 1))  # adapt this if using `channels_first` image data format
    xTest = np.reshape(xTest, (len(xTest), 28, 28, 1))  # adapt this if using `channels_first` image data format

    noiseFactor = 0.4
    xTrain_noisy = xTrain + noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=xTrain.shape)
    xTest_noisy = xTest + noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=xTest.shape)

    xTrain_noisy = np.clip(xTrain_noisy, 0., 1.)
    xTest_noisy = np.clip(xTest_noisy, 0., 1.)

    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    X = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = Conv2D(32, (3, 3), activation='relu', padding='same')(X)
    Encoded = MaxPooling2D((2, 2), padding='same')(X)

    # at this point the representation is (7, 7, 32)

    X = Conv2D(32, (3, 3), activation='relu', padding='same')(Encoded)
    X = UpSampling2D((2, 2))(X)
    X = Conv2D(32, (3, 3), activation='relu', padding='same')(X)
    X = UpSampling2D((2, 2))(X)
    Decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(X)


    Auto_Encoder = Model(input_img, Decoded)
    Auto_Encoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    history = Auto_Encoder.fit(xTrain_noisy, xTrain,
                               epochs=100,
                               batch_size=128,
                               shuffle=True,
                               validation_data=(xTest_noisy, xTest))

    Auto_Encoder.save('./')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    Denoised_Img = Auto_Encoder.predict(xTest)

    plt.figure(figsize=(20, 2))
    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i+1)
        plt.imshow(xTest_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 10, i + 10)
        plt.imshow(Denoised_Img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()

    print("Press 1 to  show Images from Original Data-set")
    print("Press 2 to see Gaussian Distribution based noise generated data-set")
    print("Press 4 to Exit")
    ch = int(input("Enter Your Choice:"))
    
    try:
    
        if ch == 1:
            plt.figure(figsize=(20, 2))
            for i in range(10):
                ax = plt.subplot(1, 10, i + 1)
                plt.imshow(xTest[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.show()

        elif ch == 2:
            plt.figure(figsize=(20, 2))
            for i in range(10):
                ax = plt.subplot(1, 10, i+1)
                plt.imshow(xTest_noisy[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

    except:
        print("Inavlid Input")

main()