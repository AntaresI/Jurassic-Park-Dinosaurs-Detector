import os
from PIL import Image
import shutil

with open("Dinosaurs.txt","r") as dinos:
      dinos_list = tuple(dinos.read().split("\n"))
      
png_dir = "Jurassic Park Dinosaurs Dataset/png"     
for i in range(len(dinos_list)):
  
    os.makedirs(png_dir+'/'+dinos_list[i],exist_ok=True)
    
    directory = "Jurassic Park Dinosaurs Dataset/"+dinos_list[i] 
    c=1
    for filename in os.listdir(directory):
        if not filename.endswith(".png"):
            im = Image.open(directory+'/'+filename)
            name='img'+str(c)+'.png'
            rgb_im = im.convert('RGB')
            rgb_im.save(name)
            shutil.move(name,png_dir+'/'+dinos_list[i])
            c+=1
            print(os.path.join(directory, filename))
            continue
        else:
            continue
        
        
        
        
