from bing_image_downloader import downloader
import os
import time
import urllib.request
from selenium.webdriver.common.keys import Keys



with open("Dinosaurs.txt","r") as dinos:
      dinos_list = tuple(dinos.read().split("\n"))

dir_name = "Jurassic Park Dinosaurs Dataset"
os.makedirs(dir_name,exist_ok=True)


'''BING DOWNLOADER'''      
for i in range(len(dinos_list)):
    
    print(f"Downloading images for {dinos_list[i]}")
    downloader.download(dinos_list[i]+" dinosaur", limit=100, adult_filter_off=True, output_dir=dir_name, force_replace=False, timeout=60, verbose=True)
''''''''''''''''''''    
    

'''GOOGLE DOWNLOADER'''

    
    
    