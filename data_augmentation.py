# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:12:03 2019

@author: Noman
"""

# example of random rotation image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import os.path


#list of training files
directoryread = "F:\lyme-disease\\processed_dataset\\train\\EM\\"
directorysave = "F:\lyme-disease\\processed_dataset\\augmentation\\train\\EM\\"   
imagespath = []
try:
            #exists function work for both file and directory
    if os.path.exists(directoryread):
        print("\n\nDirectory Discovered!\n\n")
        imagespath = os.listdir(directoryread)
    else:
        print("\n\nDirectory Not Found!\n\n")
except FileNotFoundError:
    print("File not Found Error!")
finally:
    print("\nTotal Files:" + str(len(imagespath)))
            
    
print(imagespath)


datagen = ImageDataGenerator(
        rotation_range= 90,
        width_shift_range=[0.2,1.0],
        height_shift_range=[0.2,1.0],
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')


images_array = []
for path in imagespath:
    img = load_img( directoryread + path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    images_array.append(x)
    
    
pathindex = 0    
for ia in images_array:    
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(ia, 
                              batch_size=1,
                              save_to_dir = directorysave, 
                              save_prefix= imagespath[pathindex], 
                              save_format='jpg'):
        i += 1
        if i > 10:
            break  # otherwise the generator would loop indefinitely
            
    pathindex = pathindex + 1