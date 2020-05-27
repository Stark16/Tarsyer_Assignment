from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import model_from_json, Sequential, Model
from keras.optimizers import Adagrad


print("\n\t Creating the final Model-")
print("\t Importing Auto-Encoder Model:")


try:
    file = open("./Trained_Models/Auto_Encoder_Trained_Model.json","r")
    model = file.read()
    file.close()
    Auto_Encoder = model_from_json(model)
    Auto_Encoder.load_weights("./Trained_Models/Auto_Encoder.h5")

    print("\t Auto_Encoder Loaded Successfully")
except:
    print("\t An Error occurred While loading Auto_Encoder Model.")

print("\n\t Importing Classifier Model:")

try:
    file = open("./Trained_Models/Classifier_Trained_Model.json", "r")
    model = file.read()
    file.close()
    Classifier = model_from_json(model)
    Classifier.load_weights("./Trained_Models/Classifier_Model.h5")

    print("\t Classifier Loaded successfully")
except:
    print("\t An Error occurred While loading Classifier Model.")

# Concatenating the Models:

print("\n\t Now Concatenating the Models:")

combine = Concatenate([Auto_Encoder, Classifier])

Final_Model = Model(input(Auto_Encoder), combine)
ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
Final_Model.compile(optimizer=ada_grad, loss='binary_crossentropy',
               metrics=['accuracy'])

print("\n\t Model Concatenation Successful")



