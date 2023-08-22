#powershell Start-Process powershell -Verb runAs
import random #picking random response
import pickle #used for converting python objects to byte code
import json 

import numpy as np

import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer


intents = json.loads(open("intents.json").read())

words = []
classes  = []
documents = []
ignore_letters = ['?', '!', '.', ','] #these are characters to ignore

for intent in intents['intents']:
    
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.append(word_list)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatized_words = []
for word in words:
    if word not in ignore_letters:
        lemmatized_words.append(lemmatizer.lemmatize(word))

words = lemmatized_words






