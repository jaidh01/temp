from flask import Flask, request, jsonify, render_template
import random
import json
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import pickle
from nltk.stem.lancaster import LancasterStemmer

app = Flask(__name__)

stemmer = LancasterStemmer()

# Load intents file
with open("intents[1].json") as file:
    data = json.load(file)

# Load model and data
try:
    with open("data_2.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # Handle the error or re-train the model here if needed
    pass

# Define the model
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

# Load the trained model
model.load("xana.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chat(inp):
    results = model.predict([bag_of_words(inp, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    return random.choice(responses)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_text = request.args.get('msg')
    return chat(user_text)

if __name__ == "__main__":
    app.run(debug=True)
