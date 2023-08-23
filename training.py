#powershell Start-Process powershell -Verb runAs
import random #picking random response
import pickle #used for converting python objects to byte code
import json 

import numpy as np

import nltk
nltk.download('punkt')
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer #used for making words into dictionary root form
#ex. running runs ran -> run

from keras.models import Sequential #layered Network
from keras.layers import Dense, Activation, Dropout 
#Dense is a fully connected layer
#Activation has functions that can be applied to layer outputs
#Dropout is used to help prevent overfitting

from keras.optimizers import Adam #Stochastic Gradient Descent used to update parameters during training

lemmatizer = WordNetLemmatizer()


intents = json.loads(open("intents.json").read()) #get intents from json

words = [] #stores tokenized words
classes  = [] #stores intents
documents = [] #stores (tokenized word, intent)
ignore_letters = ['?', '!', '.', ','] #these are characters to ignore

for intent in intents['intents']: #loop through intents
    for pattern in intent['patterns']: #loop through patterns
        word_list = nltk.word_tokenize(pattern) #tokenize the pattern
        words.extend(word_list) #add that to words
        documents.append((word_list, intent['tag'])) #add tthe word and intent

        if intent['tag'] not in classes: #if an intent is not in classes then add it
            classes.append(intent['tag']) 

#lemmatizes the words in words and updates the list with the lemmatized words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb')) #pickles words into words.pl
pickle.dump(classes, open('classes.pkl', 'wb')) #pickles classes into classes.pl

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []

    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#sgd = SGD(learning_rate=.01, momentum=.9, nesterov=True)
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)
model.save('chatbot_model.h5', hist)
print("Done")






