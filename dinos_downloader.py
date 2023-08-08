from bing_image_downloader import downloader
import os

import shutil
from PIL import Image
import imagehash
import numpy as np

'''LOADING NAMES OF DINOSAURS AND PREPARING THE MAIN DIRECTORY'''
with open("Dinosaurs.txt","r") as dinos:
      dinos_list = tuple(dinos.read().split("\n"))

dir_name = "Jurassic Park Dinosaurs Dataset"
os.makedirs(dir_name,exist_ok=True)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''CLASS THAT REMOVES DUPLICATE IMAGES FROM A FOLDER'''
class DuplicateRemover:
    def __init__(self,dirname,hash_size = 8):
        self.dirname = dirname
        self.hash_size = hash_size
        
    def find_duplicates(self):
        """
        Find and Delete Duplicates
        """
        
        fnames = os.listdir(self.dirname)
        hashes = {}
        duplicates = []
        print("Finding Duplicates Now!\n")
        for image in fnames:
            with Image.open(os.path.join(self.dirname,image)) as img:
                temp_hash = imagehash.average_hash(img, self.hash_size)
                if temp_hash in hashes:
                    print("Duplicate {} \nfound for Image {}!\n".format(image,hashes[temp_hash]))
                    duplicates.append(image)
                else:
                    hashes[temp_hash] = image
                   
        if len(duplicates) != 0:
          
            space_saved = 0
           
            for duplicate in duplicates:
                    space_saved += os.path.getsize(os.path.join(self.dirname,duplicate))
                    
                    os.remove(os.path.join(self.dirname,duplicate))
                    print("{} Deleted Succesfully!".format(duplicate))
    
        
        else:
            print("No Duplicates Found :(")
''''''''''''''''''''''''''''''''''''''''''''''''''''''            


'''CREATION OF DATASET BY EXECUTING 2 POSSIBLE BING QUERIES FOR EACH DINOSAUR, MERGING OF THE 2 IMAGE FOLDERS AND REMOVING DUPLICATES'''      
for i in range(len(dinos_list)):
    
    print(f"Downloading images for {dinos_list[i]}")
    downloader.download(dinos_list[i]+" dinosaur", limit=100, adult_filter_off=True, output_dir=dir_name, force_replace=False, timeout=60, verbose=True)
    downloader.download(dinos_list[i], limit=100, adult_filter_off=True, output_dir=dir_name, force_replace=False, timeout=60, verbose=True)
 
  
    dire = dir_name+'\\'+dinos_list[i]
    
    for file in os.listdir(dire): 
        old_filepath = os.path.join(dire, file)
        name, ext = os.path.splitext(file)
       
        new_name = name + ' 2nd query'
        new_name = new_name+ext
        new_filepath = os.path.join(dire, new_name)
    
        os.rename(old_filepath, new_filepath)


    current_folder = os.getcwd() 
      
    # list of folders to be merged
    list_dir = [dir_name+'\\'+dinos_list[i]+" dinosaur"]
      
    # enumerate on list_dir to get the 
    # content of all the folders ans store it in a dictionary
    content_list = {}
    for index, val in enumerate(list_dir):
        path = os.path.join(current_folder, val)
        content_list[ list_dir[index] ] = os.listdir(path)

    merge_folder = dir_name+'\\'+dinos_list[i]
  
    #    merge_folder path - current_folder 
    # + merge_folder
    merge_folder_path = os.path.join(current_folder, merge_folder) 
    
    for sub_dir in content_list:
  
    # loop through the contents of the
    # list of folders
        for contents in content_list[sub_dir]:
          
        # make the path of the content to move 
            path_to_content = sub_dir + "\\" + contents  
  
        # make the path with the current folder
            dir_to_move = os.path.join(current_folder, path_to_content )
  
        # move the file
            shutil.move(dir_to_move, merge_folder_path)
    os.rmdir(list_dir[0])
    
    

    # Remove Duplicates
    dr = DuplicateRemover(dire)
    dr.find_duplicates()
''''''''''''''''''''''''     
