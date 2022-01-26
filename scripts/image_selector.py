import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import psutil
import time
import cv2
import os


DATA_PATH = '/home/abdelrahman/Uni/Thesis/data/img'

# f = open("demofile2.txt", "a")
# f.write("Now the file has more content!")
# f.close()
PROCESSED_FILE = open('processed.txt','r')

PROCESSE_IMAGES_NAMES = []

PROCESSE_IMAGES_NAMES = PROCESSED_FILE.read()

cur = 0
for image_path in os.listdir(DATA_PATH):
	input_path = os.path.join(DATA_PATH, image_path)
	curr_name = image_path
	if(image_path in PROCESSE_IMAGES_NAMES):
		continue

	PROCESSED_FILE = open('processed.txt',"a")
	PROCESSED_FILE.write(input_path + '\n')
	PROCESSED_FILE.close()
	
	cv2.imshow('name',cv2.imread(input_path))
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
	
	choice = input("If you want to save this image name, press (s)..\n")
	if(choice == 's'):
		f = open('image_list.txt',"a")
		f.write(input_path + '\n')
		f.close()
