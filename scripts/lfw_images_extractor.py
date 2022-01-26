import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import psutil
import time
import cv2
import os

'''
This script aims to extract the images from it's folders in LFW dataset and save it to known_people folder
'''


DATA_PATH = '/home/abdelrahman/Uni/Thesis/lfw'

KNOWN_PEOPLE_PATH = '/home/abdelrahman/Uni/Thesis/known_people/'

for subdir, dirs, files in os.walk(DATA_PATH):
	for file in files:
		print(os.path.join(subdir, file))
		img = cv2.imread(os.path.join(subdir, file))
		cv2.imwrite(KNOWN_PEOPLE_PATH + file, img)

#SCRIPTS = '/home/abdelrahman/Uni/Thesis/scripts/'

# for subdir, dirs, files in os.walk(SCRIPTS):
# 	for file in files:
# 		if(os.path.join(subdir, file)[-3:]=='jpg'):
# 			os.remove(file)