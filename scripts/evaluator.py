'''
This script aims to evaluate the face detector based on the annotations done using annotator.py

The following needs to be implemented:
 - load the annotatoins file
 - For each image:
   . detect all the faces in it.
   . for each face:
     .calculate the iou between that face and all the face in the annotation file for this image "id is unique"
     .get the closest box and compare the corresponing name with the current faca name.
     .increment the number of faces by 1
     .if the detection of the person was correct,incemrement the number of correct predictoins by one. otherwise increment the number of false predictions by one
'''

import face_recognition
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import psutil
import time
import cv2
import os
import json
import pickle


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


with open('annotations.json', 'r') as fp:
    annotations = json.load(fp)

CORRECT_PREDICTIONS = 0.0
NUMBER_OF_FACES = 0.0
FALSE_PREDICTIONS = 0.0

KNOWN_PEOPLE_PATH = '../known_people/'
TEST_PATH = '../test/'
# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

# Load a sample picture and learn how to recognize it.

UPDATE_LIST = True

if(UPDATE_LIST == True):

	with open("known_face_encodings.txt", "rb") as new_filename:
		known_face_encodings = pickle.load(new_filename)

	with open("known_face_names.txt", "r") as f:
		for line in f:
  			known_face_names.append(line.strip())



else:

	for image_path in os.listdir(KNOWN_PEOPLE_PATH):
		input_path = os.path.join(KNOWN_PEOPLE_PATH, image_path)
		curr_name = image_path[:len(image_path)-4]
		curr_image = face_recognition.load_image_file(input_path)
		try:
			curr_image_face_encoding = face_recognition.face_encodings(curr_image, model='large')[0]
			known_face_encodings.append(curr_image_face_encoding)
			known_face_names.append(curr_name)
		except:
			print('IN CATCH')
			print(curr_name)

	with open("known_face_encodings.txt", "wb") as internal_filename:
		pickle.dump(known_face_encodings, internal_filename)

	with open("known_face_names.txt", "w") as f:
		for s in known_face_names:
			f.write(str(s) +"\n")
	
print('first step done!')

for image_path in os.listdir(TEST_PATH):

	print('In the loop..')
	input_path = os.path.join(TEST_PATH, image_path)
	unknown_image = face_recognition.load_image_file(input_path)

	print(input_path)
	# Find all the faces and face encodings in the unknown image
	face_locations = face_recognition.face_locations(unknown_image, model="cnn")#, model="cnn"
	face_encodings = face_recognition.face_encodings(unknown_image, face_locations, model='large')

	# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
	# See http://pillow.readthedocs.io/ for more about PIL/Pillow
	pil_image = Image.fromarray(unknown_image)
	# Create a Pillow ImageDraw Draw instance to draw with
	draw = ImageDraw.Draw(pil_image)

	# Loop through each face found in the unknown image
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
	    # See if the face is a match for the known face(s)
	    print(left, top, right, bottom)
	    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
	    NUMBER_OF_FACES+=1
	    name = "Unknown"

	    # If a match was found in known_face_encodings, just use the first one.
	    # if True in matches:
	    #     first_match_index = matches.index(True)
	    #     name = known_face_names[first_match_index]

	   # Or instead, use the known face with the smallest distance to the new face
	    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
	    best_match_index = np.argmin(face_distances)
	    if matches[best_match_index]:
	        name = known_face_names[best_match_index]
	        if(face_distances[best_match_index] > 0.5):
	        	name = "Unknown"
	        print('Best_match_index:',face_distances[best_match_index])
	        print('Name:',known_face_names[best_match_index])

	    curr_image_annotation = annotations[image_path]
	    max_intersection = 0.0
	    corresponding_name = ''
	    # print(curr_image_annotation)
	    # print(curr_image_annotation[0])
	    # print(curr_image_annotation[0][1])
	    #[left, top, right, bottom]
	    #{'x1', 'x2', 'y1', 'y2'}
	    for i in range(len(curr_image_annotation)):
	    	left_2 = curr_image_annotation[i][1][0]
	    	right_2 = curr_image_annotation[i][1][2]
	    	top_2 = curr_image_annotation[i][1][1]
	    	bottom_2 = curr_image_annotation[i][1][3]
	    	intersection = bb_intersection_over_union([left, top,right, bottom],[left_2, top_2, right_2, bottom_2])
	    	if(intersection > max_intersection):
	    		max_intersection = intersection
	    		corresponding_name = curr_image_annotation[i][0]

	    print(name, ' ', corresponding_name, ' ', max_intersection)
	    if(corresponding_name == name):
	    	CORRECT_PREDICTIONS+=1
	    else:
	    	FALSE_PREDICTIONS+=1

print(NUMBER_OF_FACES)
print(CORRECT_PREDICTIONS)
print(FALSE_PREDICTIONS)

print('Accuracy = ', (CORRECT_PREDICTIONS/NUMBER_OF_FACES)*100,'%')