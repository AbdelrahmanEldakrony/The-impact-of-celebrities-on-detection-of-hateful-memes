import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import psutil
import time
import cv2
import os
import json
import pickle

KNOWN_PEOPLE_PATH = '../known_people/'
TEST_PATH = '../test/'
HATEFUL_MEMES_DATASET_PATH = '../data_12k/hateful_memes/img_clean/'
# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []


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

celeb_boxes = {}


for image_path in os.listdir(HATEFUL_MEMES_DATASET_PATH):

	print('In the loop..')
	input_path = os.path.join(HATEFUL_MEMES_DATASET_PATH, image_path)
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

	#It will contain all bounding boxes for all celebs in the image
	boxes = []
	#It will contain all the names of the corresponding box
	names = []

	# Loop through each face found in the unknown image3
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
	    # See if the face is a match for the known face(s)
	    print(left, top, right, bottom)
	    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

	    name = "Unknown"

	   
	    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
	    best_match_index = np.argmin(face_distances)
	    if matches[best_match_index]:
	        name = known_face_names[best_match_index]
	        if(face_distances[best_match_index] > 0.5):
	        	name = "Unknown"
	        print('Best_match_index:',face_distances[best_match_index])
	        print('Name:',known_face_names[best_match_index])

	    if(name != "Unknown"):
	    	curr_box = [top, right, bottom, left]
	    	boxes.append(curr_box)
	    	name = name.split('_')
	    	name_string = ''
	    	for i in name:
	    		if(i.isnumeric()):
	    			pass
	    		else:
	    			name_string+=i
	    			name_string+=' '
	    	name_string = name_string.lower()
	    	if(name_string[len(name_string)-1]== ' '):
	    		name_string = name_string[:-1]
	    	names.append(name_string)

	    # Draw a box around the face using the Pillow module
	    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

	    # Draw a label with a name below the face
	    text_width, text_height = draw.textsize(name)
	    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
	    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

	    #put the follwoing out in case of testing, now is for annotating
	    


	# Remove the drawing library from memory as per the Pillow docs
	
	newDic = dict()
	newDic['celeb_boxes'] = boxes
	newDic['names'] = names
	celeb_boxes[image_path] = newDic
	

with open('celeb_boxes.json', 'w') as fp:
		json.dump(celeb_boxes, fp)

	

