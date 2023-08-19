
# from PIL import Image
# from keras.models import load_model
# import numpy as np
# import os
# import json
  



def predict_one(img_path):
    from PIL import Image
    from keras.models import load_model
    import numpy as np
    #import os
    import json
    print(img_path)
    if type(img_path) != str:
        raise TypeError("Path to image must be a string")
        
    with open('dino_dict.txt') as f:
        dino_labels = f.read()
        
    dino_labels = json.loads(dino_labels) 
   
    model = load_model('mobilenet_weights_1.h5')
    
    im = Image.open(img_path)

    imageresize = im.resize((224,224))
    
    imageresized = np.asarray(imageresize)
    img_tensor = np.expand_dims(imageresized, axis=0)
    print(model.predict(img_tensor))
    prediction = np.argmax(model.predict(img_tensor))
    
    for keys,values in dino_labels.items():
        if values == prediction:
              name = keys
    return name        
                