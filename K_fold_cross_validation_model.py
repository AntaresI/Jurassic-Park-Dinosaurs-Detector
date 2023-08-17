# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:15:17 2023

@author: Johnny
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.models import Model
#from tensorflow.keras.applications import EfficientNetV2S
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.applications import InceptionV3
#from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import json

def create_model():
    
    """LOADING A PRE-TRAINED MODEL FOR TRANSFER LEARNING"""
    vgg = VGG16(input_shape=[224,224,3], weights='imagenet', include_top=False)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """UNFREEZING THE MODEL SO THAT THE WEIGHTS CAN CHANGE (WORKED THE BEST AFTER SEVERAL EXPERIMENTS)"""
    for layer in vgg.layers[:4]:
        layer.trainable = False
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """ADDING A DENSE LAYER WITH DROPOUT AT THE END OF THE MODEL AND THEN PREDICTION LAYER"""
    x = Flatten()(vgg.output)
    x = Dense(700, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction = Dense(47, activation='softmax')(x)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """INSTANTIATE THE MODEL"""
    model = Model(inputs=vgg.input, outputs=prediction)
    #print(len(model.layers))
    """"""""""""""""""""""""""
    
    """VIEW THE MODEL LAYERS AND PROPERTIES"""
    model.summary()
    """"""""""""""""""""""""""""""""""""""""""
    
    """CREATING THE OPTIMIZER"""
    # learning_rate_decay_factor = (0.0001/0.001)
    # decay = tf.optimizers.schedules.CosineDecay(initial_learning_rate=0.001,decay_steps=args.epochs*(10000/args.batch_size), alpha=learning_rate_decay_factor)
    optimize = tf.optimizers.Adam(learning_rate = 3e-5)
    #optimize = tf.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
    """"""""""""""""""""""""""""""
    
    
    """COMPILING MODEL, TRAINING WITH CATEGORICAL CROSSENTROPY"""
    model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer= optimize,
      metrics=['accuracy']
    )
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    return model


if __name__ == "__main__":
    
    """INITIALIZING A GENERATOR FOR IMAGES WITH SOME PREPROCESSING"""
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                  
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                        horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    val_datagen = ImageDataGenerator(rescale = 1./255)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    """ARGUMENT PARSING FOR EASY CALLING OF SOME PARAMETERS"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
    
    args = parser.parse_args([] if "__file__" not in globals() else None)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """PATHS TO TRAIN VAL AND TEST FOLDERS"""
    train_path = "Train"
    test_path = "Test"
    val_path = "Val"
    """"""""""""""""""""""""""""""""""""""""""
    
    """LOADING THE IMAGE SETS FOR TRAINING, VALIDATION AND TESTING"""
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size = (224, 224),
                                                     batch_size = args.batch_size,
                                                     class_mode = 'sparse')
    
    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size = (224, 224),
                                                batch_size = args.batch_size,
                                                class_mode = 'sparse')
    
    val_set = val_datagen.flow_from_directory(val_path,
                                                target_size = (224, 224),
                                                batch_size = args.batch_size,
                                                class_mode = 'sparse')
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """CONCATENATING TRAIN AND VAL DATASET FOR SPLITTING INTO K FOLDS"""
    
    train_images=np.concatenate([training_set.next()[0] for i in range(training_set.__len__())])
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size = (224, 224),
                                                     batch_size = args.batch_size,
                                                     class_mode = 'sparse')
    train_labels=np.concatenate([training_set.next()[1] for i in range(training_set.__len__())])
    
    val_images=np.concatenate([val_set.next()[0] for i in range(val_set.__len__())])
    
    val_set = val_datagen.flow_from_directory(val_path,
                                                target_size = (224, 224),
                                                batch_size = args.batch_size,
                                                class_mode = 'sparse')
    val_labels=np.concatenate([val_set.next()[1] for i in range(val_set.__len__())])
    
    inputs = np.concatenate((train_images, val_images), axis=0)
    targets = np.concatenate((train_labels, val_labels), axis=0)
    
    kfold = KFold(n_splits=5, shuffle=True)
    counter = 0
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    """SAVING INFORMATION ABOUT HOW THE LABELS CORRESPOND TO EACH DINOSAUR"""  
    with open('dino_dict.txt', 'w') as convert_file:
         convert_file.write(json.dumps(training_set.class_indices))
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    
    
    """PERFORMING THE K-FOLD TRAINING"""
    for train, test in kfold.split(inputs, targets):
    
        counter += 1
        """LOADING A PRE-TRAINED MODEL FOR TRANSFER LEARNING"""
        
        model = create_model()
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        """FIT THE MODEL"""
        history = model.fit(
          inputs[train],
          targets[train],
          epochs=args.epochs,
          batch_size=args.batch_size,shuffle=True)
        """"""""""""""""""""
        
        """PREDICTING ON THE TEST SET"""
        model.evaluate(inputs[test],targets[test],batch_size=args.batch_size)
        """"""""""""""""""""""""""""""""
        
        """SAVING WEIGHTS"""
        model.save("vgg16 3-weights-final{counter}.h5")
        """"""""""""""""""""
        
        """PLOTTING THE ACCURACY AND LOSS AFTER TRAINING"""
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.show()
        plt.plot(history.history['accuracy'], label='train accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy [%]")
        plt.legend()
        
        plt.show()
        """"""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""