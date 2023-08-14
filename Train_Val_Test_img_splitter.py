
import os
import shutil

"""LOAD THE DINOSAUR NAMES"""
with open("Dinosaurs.txt","r") as dinos:
      dinos_list = tuple(dinos.read().split("\n"))
""""""""""""""""""""""""""

"""MAKE TRAIN, VAL AND TEST FOLDERS"""
dirs = ["Train", "Val", "Test"]

for dirs in dirs:
    
     os.makedirs(dirs,exist_ok=True)

data_dir = "Jurassic Park Dinosaurs Dataset"
""""""""""""""""""""""""""""""""""""""""""




def train_val_test_splitter(train_split=0.64, val_split=0.16, test_split=0.20):

    """ERROR HANDLING"""    

    if type(train_split) != float or type(val_split) != float or type(test_split) != float:
        raise TypeError("Arguments must be float")
    
    if train_split+val_split+test_split != 1.0:
        raise ValueError("Sum of arguments must be equal to 1.0")         
    """"""""""""""""""


    """LOOP FOR EACH DINOSAUR"""        
    for i in range(len(dinos_list)):
        
        
        """FIND OUT HOW MANY IMAGES OF THAT DINOSAUR ARE THERE IN THE DATASET""" 
        img_directory = data_dir+'/png/'+dinos_list[i]
        image_len = len(os.listdir(img_directory))
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        """GET THE NUMBER OF IMAGES CORRESPONDING TO THE USER REQUESTED RATIO"""        
        train_len = int(image_len*train_split)
        val_len = int(image_len*val_split)
        test_len = int(image_len*test_split)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        """THE NUMBERS WON'T ADD UP TO THE NUMBER OF ALL IMAGES OF THAT DINOSAUR, SO ADD ONE TO TRAIN UNTIL IT CORRESPONDS"""    
        while True:
            if train_len+val_len+test_len != image_len:
                train_len += 1
            if train_len+val_len+test_len == image_len:
                break
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        """CREATING THE FOLDERS THAT WILL CONTAIN IMAGES OF THAT DINO FOR TRAIN, VAL AND TEST"""        
        img_dir_images = os.listdir(img_directory)
        
        tvt_directories = ['Train/'+dinos_list[i],'Val/'+dinos_list[i],'Test/'+dinos_list[i]] #train val test directories 
        for directories in tvt_directories:
            os.makedirs(directories,exist_ok=True)
        
        for j in range(len(img_dir_images)):
            
             if j<train_len:
                 shutil.copy(img_directory+'/'+img_dir_images[j],tvt_directories[0])
                     
             elif j<train_len+val_len:
            
                 shutil.copy(img_directory+'/'+img_dir_images[j],tvt_directories[1])
                 
             else:  
                 shutil.copy(img_directory+'/'+img_dir_images[j],tvt_directories[2])
              
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
train_val_test_splitter()   
            