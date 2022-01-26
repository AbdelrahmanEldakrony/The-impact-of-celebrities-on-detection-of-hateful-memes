import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import psutil
import time
import cv2
import os


IMAGE_LIST_FILE = open('/home/abdelrahman/Uni/Thesis/annotation_tool_demo/image_list.txt','r')
#'/home/abdelrahman/Uni/Thesis/scripts/image_list.txt'
IMAGE_LIST = []

IMAGE_LIST = IMAGE_LIST_FILE.readlines()


for i in IMAGE_LIST:

	i_list = i.strip().split('/')
	print(i_list[len(i_list)-1][:-4])
	#img = cv2.imread('r'+i)
	img = cv2.imread(i.strip())
	cv2.imwrite('/home/abdelrahman/Uni/Thesis/annotation_tool_demo/img/' + str(i_list[len(i_list)-1][:-4]) + '.png', img)

#'/home/abdelrahman/Uni/Thesis/test/'

