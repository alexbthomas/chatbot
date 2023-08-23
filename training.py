#powershell Start-Process powershell -Verb runAs
import random #picking random response
import pickle #used for converting python objects to byte code
import json 

import numpy as np

import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer #used for making words into dictionary root form
#ex. running runs ran -> run

from keras.models import Sequential #layered Network
from keras.layers import Dense, Activation, Dropout 
#Dense is a fully connected layer
#Activation has functions that can be applied to layer outputs
#Dropout is used to help prevent overfitting

from keras.optimizers import SGD #Stochastic Gradient Descent used to update parameters during training

lemmatizer = WordNetLemmatizer


intents = json.loads(open("intents.json").read()) #get intents from json

words = [] #stores tokenized words
classes  = [] #stores intents
documents = [] #stores (tokenized word, intent)
ignore_letters = ['?', '!', '.', ','] #these are characters to ignore

for intent in intents['intents']: #loop through intents
    for pattern in intent['patterns']: #loop through patterns
        word_list = nltk.word_tokenize(pattern) #tokenize the pattern
        words.append(word_list) #add that to words
        documents.append((word_list, intent['tag'])) #add tthe word and intent

        if intent['tag'] not in classes: #if an intent is not in classes then add it
            classes.append(intent['tag']) 

#lemmatizes the words in words and updates the list with the lemmatized words
lemmatized_words = []
for word in words:
    if word not in ignore_letters:
        lemmatized_words.append(lemmatizer.lemmatize(word))

words = lemmatized_words
print(words)






