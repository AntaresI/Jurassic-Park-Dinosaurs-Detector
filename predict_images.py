

from PIL import Image
from keras.models import load_model
import numpy as np
import os
import json
  
# reading the data from the file
with open('dino_dict.txt') as f:
    dino_labels = f.read()
  

      
# reconstructing the data as a dictionary
dino_labels = json.loads(dino_labels)
  
model = load_model('Weights/mobilenet_weights_1.h5')

directory = "images_to_predict"


for filename in os.listdir(directory):
   
        im = Image.open(directory+'/'+filename)
      
        image = np.asarray(im)

        imageresize = im.resize((224,224))
        
        imageresized = np.asarray(imageresize)
        img_tensor = np.expand_dims(imageresized, axis=0)
        
        prediction = np.argmax(model.predict(img_tensor))
        
        for keys,values in dino_labels.items():
            if values == prediction:
                print(f"Model predicts that this is a {keys}")
        
        