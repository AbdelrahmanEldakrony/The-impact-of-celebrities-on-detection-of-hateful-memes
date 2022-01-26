'''
This script aims to annotate the images in the testing folder to test the face detector.

The following needs to be implemented:
.For each person in each image append the following to the json file: 
  -name of the person
  -bouding box coordinates

returns annotations.json
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

# dictionary_data = {"a": [['Barak Obama',[1,2,3,4]],['Barak Obama',[1,2,3,4]],['Barak Obama',[1,2,3,4]]], "b": ['Donald Trump ',[4,5,6,7]]}

# dictionary_data['c'] = []
# dictionary_data['c'].append(['t1',[0,0,0,0]])
# dictionary_data['c'].append(['t2',[2,2,2,2]])
# with open('data.json', 'w') as fp:
#     json.dump(dictionary_data, fp)


# with open('data.json', 'r') as fp:
#     data = json.load(fp)

# print(data['c'][1])
# exit(0)

def annotate():
	annotations = dict()
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

	# This is an example of running face recognition on a single image
	# and drawing a box around each person that was identified.

	# Load an image with an unknown face

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
		    pil_image = Image.fromarray(unknown_image)
		    draw = ImageDraw.Draw(pil_image)
		    # Draw a box around the face using the Pillow module
		    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

		    # Draw a label with a name below the face
		    text_width, text_height = draw.textsize(name)
		    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
		    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
		    
		    if(image_path not in annotations):
		    	annotations[image_path] = []
		    
		    celeb_name = input('Enter celeb name: ')
		    annotations[image_path].append([celeb_name,[left, top, right, bottom]])
		    #put the follwoing out in case of testing, now is for annotating
		    del draw
		    pil_image.show()
		    time.sleep(1)
		    for proc in psutil.process_iter():
		    	if proc.name() == "display":
		    		proc.kill()


	with open('annotations.json', 'w') as fp:
		json.dump(annotations, fp)

with open('annotations.json', 'r') as fp:
    data = json.load(fp)

print(len(data.keys()))

# print(data['70158.png'])


with open('celeb_boxes_10k.json', 'r') as fp:
    data = json.load(fp)

# print(list(data.items())[0:15])
print('Out from celeb_boxes_10k: ',data['04971.png'])
# Out from celeb_boxes_10k:  {'celeb_boxes': [[218, 343, 511, 50]], 'names': ['nicolas cage']}

with open('celeb_graph_knowledge.json', 'r') as fp:
    data = json.load(fp)

print(data['nicolas cage'])
# print(len(data.keys()))

# print(list(data.items())[0:15])

# x = list(data.items())[0:5]



# for key, value in data.items():
# 	curr_names = []
# 	for value2 in value['names']:
# 		if(value2 == 'adolf hitler'):
# 			print('hey')
	# 	else:
	# 		curr_names.append(value2)
	# value['names'] = curr_names

# with open('celeb_boxes_2.json', 'w') as fp:
# 		json.dump(data, fp)
# celeb_boxes_test = {}

# k = 'a'

# boxes = [[1,2,3,4],[0,0,0,0],[5,5,5,5,]]
# names = ['a','b','c']

# newDic = dict()

# newDic['celeb_boxes'] = boxes
# newDic['names'] = names
# celeb_boxes_test['a'] =newDic
