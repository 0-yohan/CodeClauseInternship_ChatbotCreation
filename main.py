import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import tensorflow
# import tflearn
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pickle
from tensorflow.keras.models import load_model

stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    ignore_words = ['?']

    for intent in data["intents"]:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            docs_x.append(tokens)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output  = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

# tensorflow.compat.v1.reset_default_graph()

model = Sequential([
Dense(8, activation='relu', input_shape=(len(training[0]),)),
Dense(8, activation='relu'),
Dense(len(output[0]), activation='softmax')
])

try:
   model = load_model("./model.h5")
except:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(training, output, epochs=1000, batch_size=8)
    model.save("model.h5")  


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if se == w:
                bag[i] = 1
    
    # return np.array(bag)
    return np.array(bag)[np.newaxis, :] 

def chat():
    print("Start Chatting, send 'quit' to Quit:\n")

    while True:
        inp = input("You: ")

        if inp.lower() == 'quit':
            break

        # results = model.predict([bag_of_words(inp, words)])
        results = model.predict([bag_of_words(inp, words)], verbose=0)[0]  # Set verbose=0 to supress keras logs
        result_index = np.argmax(results)
        if results[result_index] >= 0.7 :
            tag = labels[result_index]

            for tags in data["intents"]:
                if tags["tag"] == tag:
                    responses = tags["responses"]
        
            print(random.choice(responses))
        
        else:
            print("I didn't get that, Try again!")


chat()
        
