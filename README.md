# Jurassic-Park-Dinosaurs-Detector
Jurassic Park 47 dinosaurs Convolutional Neural Network (CNN) model trained with Tensorflow on images of dinosaurs from Bing.

![hi](/images_to_predict/Dimetrodon.png)
# Website
The model is hosted via Streamlit here: [Dinosaur detector](https://mqvscmjomxjrnigtvmxevz.streamlit.app/)

# Demonstration
Here is a quick demonstration of the website and model predictions
![hi](/images_to_predict/Show.png)
# Dataset
I wrote and used the script `dinos_downloader.py` which downloads images from Bing to create my own custom dataset of 47 species of dinosaurs. The dataset can be found here: [Jurassic Park Dinosaurs Dataset](https://www.kaggle.com/datasets/antaresl/jurassic-park-dinosaurs-dataset)

I cleaned the downloaded images both automatically (also in `dinos_downloader.py`) and manually (to get rid of wrong images) and the result is 4364 images of 47 different species of dinosaurs that are known to appear in the Jurassic Park movies.

# Workflow
After cleaning the images, they needed to be changed into a unified format `.png` so I wrote and used the `png_converter.py` script for converting all the images into `.png` format, and saving them accordingly into a `/png` folder containing all the folders of dinosaurs.

Then a training, validation and test set needed to be created, but so that there exist three folders `Train`, `Val` and `Test` each containing the folders of dinosaurs, so that I can easily use a Tensorflow method `tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory` which loads the dataset directly into `Train`, `Val` and `Test` iterators and also creates and names the labels, which correspond to the names of the dinosaurs. For that purpose I wrote the `train_val_test_splitter.py` which works with the `/png` in workplace.

# Model
I wrote the script `training_model.py` for loading the dataset into Tensorflow, constructing the model, training it and displaying loss and accuracy graphs.
I used the `MobileNet` model available in [Keras trained CNN models](https://keras.io/api/applications/). I allowed the model to train all but its first 3 layers so that it could adapt to the new small and multiple-class dataset, I also finetuned the model with a  `MaxPooling2D` layer, `Dense` layer and `dropout` layer. 

# Training K-fold
K-fold cross validation was used for a better training. Script `k_fold_cross_validation_model.py` implements the splitting of train and val data into 5 random shuffles and performs the training.

This model was trained for 40 epochs with a batch_size of 20. 
![hi](/images_to_predict/training_result.png)

The model achieved 57.24 % accuracy on the test_set

![hi](/images_to_predict/mobilenet_57_percent_acc.png)
![hi](/images_to_predict/mobilenet_57_percent_loss.png)
# Predicting 
I wrote `predict_one_image.py` script that takes a path to an image and predicts which dinosaur it is. 
For a folder of images the same can be done with `predict_images.py` where the image are required to be saved in the folder `images_to_predict.py`
TODO ukázky prediction

