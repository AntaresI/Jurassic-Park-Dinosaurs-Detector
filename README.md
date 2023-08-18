# Jurassic-Park-Dinosaurs-Detector
Jurassic Park 47 dinosaurs Convolutional Neural Network (CNN) model trained with Tensorflow on images of dinosaurs from Bing.

# Website
The model is hosted via Streamlit here: TODO

# Demonstration
Here is a quick demonstration of the website and model predictions TODO

# Dataset
I wrote and used the script `dinos_downloader.py` which downloads images from Bing to create my own custom dataset of 47 species of dinosaurs. The dataset can be found here: TODO

I cleaned the downloaded images both automatically (also in `dinos_downloader.py`) and manually (to get rid of nonsense images) and the result is 4364 images of 47 different species of dinosaurs that are known to appear in the Jurassic Park movies.

# Workflow
After cleaning the images, they needed to be changed into a unified format `.png` so I wrote and used the `png_converter.py` script for converting all the images into `.png` format, and saving them accordingly into a `/png` folder containing all the folders of dinosaurs.

Then a training, validation and test set needed to be created, but so that there exist three folders `Train`, `Val` and `Test` each containing the folders of dinosaurs, so that I can easily use a Tensorflow method `tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory` which loads the dataset directly into `Train`, `Val` and `Test` iterators and also creates and names the labels, which correspond to the names of the dinosaurs. For that purpose I wrote the `train_val_test_splitter.py` which works with the `/png` in workplace.

# Model
I wrote the script `training_model.py` for loading the dataset into Tensorflow, constructing the model, training it and displaying loss and accuracy graphs.
I experimented with many models, but in the end I realized that given that this model is very small and divided into many classes, as small pre-trained model as possible should be used, so I used the 2nd smallest available model in [Keras trained CNN models](https://keras.io/api/applications/) which is `MobileNet`. I allowed the model to train all but its first 3 layers so that it could adapt to the new small and multiple-class dataset, I also finetuned the model with a  `MaxPooling2D` layer, `Dense` layer and `dropout` layer. This model was trained for 40 epochs with a batch_size of 20
