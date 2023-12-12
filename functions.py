import pickle


import tensorflow as tf  # Import TensorFlow


from keras.api._v2.keras import Sequential
from keras.layers import Dense,Dropout
from keras.models import load_model

################################################################
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
import json
import random


from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

import nltk
nltk.download('wordnet')

import nltk
nltk.download("stopwords")













####################################################

file_name = ["classes.pkl","words.pkl","chatbot_model.h5"]

for i in range(0,3):    
    with open(file_name[0], 'rb') as file:
        classes = pickle.load(file)
    
    with open(file_name[1], 'rb') as file:
        words = pickle.load(file)

    model = load_model(file_name[2])


import json

# Specify the path to your JSON file
file_path1 = "intents.json"

# Open and read the JSON file
with open(file_path1, "r") as json_file:
    intents_json = json.load(json_file)


























def clean_up_sentence(sentence):
  sentence_words = nltk.tokenize.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(i.lower()) for i in sentence_words]
  return sentence_words



 
    
import numpy as np

def bow(sentence, words, show_details=True):
    # Tokenize and clean the pattern
    sentence_words = clean_up_sentence(sentence)

    # Create a bag of words - a matrix of N words (vocabulary matrix)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if the current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)


    print(np.shape(np.array(bag)))

    return(np.array(bag))



def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort results by probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


import random

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

V1 = predict_class("I'm not sure",model)


print(getResponse(V1,intents_json))