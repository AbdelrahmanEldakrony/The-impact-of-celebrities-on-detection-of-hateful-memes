from PIL import Image, ImageDraw
import numpy as np
import psutil
import time
import cv2
import os
import json
import pickle

def detect_web(path):
    """Detects web annotations given an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print('\nBest guess label: {}'.format(label.label))

    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images found:'.format(
            len(annotations.pages_with_matching_images)))

        for page in annotations.pages_with_matching_images:
            print('\n\tPage url   : {}'.format(page.url))

            if page.full_matching_images:
                print('\t{} Full Matches found: '.format(
                       len(page.full_matching_images)))

                for image in page.full_matching_images:
                    print('\t\tImage url  : {}'.format(image.url))

            if page.partial_matching_images:
                print('\t{} Partial Matches found: '.format(
                       len(page.partial_matching_images)))

                for image in page.partial_matching_images:
                    print('\t\tImage url  : {}'.format(image.url))

    if annotations.web_entities:
        print('\n{} Web entities found: '.format(
            len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print('\n\tScore      : {}'.format(entity.score))
            print(u'\tDescription: {}'.format(entity.description))

    if annotations.visually_similar_images:
        print('\n{} visually similar images found:\n'.format(
            len(annotations.visually_similar_images)))

        for image in annotations.visually_similar_images:
            print('\tImage url    : {}'.format(image.url))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

#detect_web('/home/abdelrahman/Uni/Thesis/face_test.png')

#https://www.holisticseo.digital/python-seo/knowledge-graph/#Using-Google-Knowledge-Graph-API-with-Advertools-to-Analyze-Entity-Profiles
def get_graph_knowledge(person):
    
    from advertools import knowledge_graph
    import xlsxwriter
    import pandas as pd
    key = 'AIzaSyDO9NpC5d2FbtxkeeBNXn8B2myJiLEnIa8'

    kg_df = knowledge_graph(key = key, query = person)

    go_on = True

    person_info_string = ''

    # In case the person doesn't have any available information
    if('resultScore' not in kg_df.columns):
        go_on = False


    if(go_on == True):
        kg_df =  kg_df.loc[kg_df['resultScore'] == kg_df['resultScore'].max()]
        
        relevant_keys = ['query', 'resultScore']
        
        if('result.description' in kg_df.columns):
            relevant_keys.append('result.description')

        if('result.detailedDescription.articleBody' in kg_df.columns):
            relevant_keys.append('result.detailedDescription.articleBody')

        if(relevant_keys[len(relevant_keys)-1] == 'result.description' or relevant_keys[len(relevant_keys)-1] == 'result.detailedDescription.articleBody'):
            person_info = kg_df[relevant_keys[len(relevant_keys)-1]]
            person_info_string = person_info[0]
    
    if(not person_info_string):
        person_info_string = 'No available information for ' + person
    
    return person_info_string

def add_gn_to_json():

    names = set()
    celeb_graph_knowledge = {}
    with open('celeb_boxes.json', 'r') as fp:
        data = json.load(fp)

    with open('celeb_boxes_10k.json', 'r') as fp:
        data2 = json.load(fp)

    for key, value in data.items():
        for value2 in value['names']:
            names.add(value2)

    for key, value in data2.items():
        for value2 in value['names']:
            names.add(value2)

    for name in names:
        info = get_graph_knowledge(name)
        celeb_graph_knowledge[name] = info

    with open('celeb_graph_knowledge.json', 'w') as fp:
        json.dump(celeb_graph_knowledge, fp)


with open('celeb_graph_knowledge.json', 'r') as fp:
    data = json.load(fp)
print(data['adolf hitler'])
print(data['barack obama'])



# export GOOGLE_APPLICATION_CREDENTIALS="/home/abdelrahman/Downloads/hatefulmemes-321020-4515196f976a.json"

# Add to the report

# “query” which the string we have searched for in the Knowledge Graph Search API.
# “@type” of the entities in our result data frame. not important
# “result.image.url” is the URL of the images of the entities in our result data frame. not important
# “result.image.contentUrl” is the direct image URL of the image of the entity. not important
# “result.@id” is the id number of the entity which is a quite important detail of a Holistic SEO. not important
# “result.detailedDescription.licence” is the license of the entity information source attribute.  not important
# “result.detailed.Description.articleBody” is a detailed explanation of the entity.VERY important
# “result.detailedDescription.url” is the URL of the detailed explanation of the entity. not important
# “result.@type” is the type of the entity with their profile information such as “thing, person or thing, the company”. not important
# “result.name” is the name of the entity. not important
# “result.description” is the short definition of the entity such as “Film in 2005, Musical Group, or Film Character”. VERY important
# “result.url” is the URL of the best definitive source of the entity which is also quite important for a Holistic SEO. not important
# “query_time” is the time that we have performed the function call. not important


