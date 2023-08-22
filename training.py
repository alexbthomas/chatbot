#powershell Start-Process powershell -Verb runAs
import random
import pickle
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

print(documents)






