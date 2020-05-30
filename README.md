# Tarsyer_Assignments

Hello,

This is the complete submission of the 2 assignment given to me on 26th of April.
This Readmd File consists of all the required documentation of and information regarding the project.


Overview:
  1.  All the scripts are commented and user interface is avaialble for ease in evaluation.
  2.  For Assignment 1, There are 2 scripts: 1.MNIST_Autoencoder and 2.Classifier.
      The scripts are used for training 1.Autoencoder model, and 2.fine tuning the same model to add a classifier, respectively.
  3.  For evaluating assigment 1, Running the AutoEncoder script will train a Autoencoder model and in the end plot both
      the loss curves, while also saving the model and wieghts in the Trained_model directory.
  4. Runnig the CLassifier script, will 1st check if the autoencoder model is trained or not, if trained it will import the same model
     then it will check if classifier model is saved or not, if yes it will import that too and just run a test image through the entire      system. In order to test the fine tuning of the network. You can delete the classifer trained models and weights and the Classifer 
     script will train the netwrk again and display the plots after completing the training.
     
  5.  For assignment 2, There is only one script named Re-arrange that will do everything from reading
      the xml file to displaying the correct sequence of the license plate.
 
*File Structure:
  > The 2 main Folders are labled as was instructed based on the Assignment Numbers and they consist
    of the 2 assigments within them.
  > Assigment 1 is the MNIST dataset problem statement. it has the following directory structure:
  
    + Loss Plots:
    |--Loss plot for autoencoder_model
    |--Loss plots for the classifier model
    |____________________________________
    
    + Trained Model:
    |--Auto_Encoder.json: The trained Autoencoder model.
    |--Auto_Encoder.h5: The weights associated to the trained model Auto_Encoder
    |--Classifier.json: The trained Classifier model.
    |--Classifier.h5: The weights associated to the trained model Classifier
    
    + Classification.py: The python script the fine tunes the Auto-encoder Model.
                        if both the networks are trained then one can directly run this script to get a test run
                        of the entire network. I.e: input-Noisy digit image, Ouput-The Number classified.
    + MNIST_Autoecoder.py: This script is used for training the Autoencoder model. You can run this script to train the model
                           anytime.
  *> Assignment 2 has 1 folder and one script.
   > The Folder containes the images and xml files assocoiated with them.
   > The script when executed, loops through all the xmls files and then finally display the correct order of license plate 
     charactors for each license plate.
   > The working of the script is explained using comments in the script itself.
   >File strucutre of Assignment 2:
   
     + Assignment 2:
     |--Char-detection
     |--Re-arrange.py
 
 
 Additional Information:
  The entire project is a Pycharm project created in windows 10, with conda environment (version: 4.7.12)
  Libraries used: 
    1. Tensorflow-gpu 1.14
    2. Keras 2.3.1
    3. xml-etree
    4. os
    5. matplot-lib
    6. numpy

    
