from flask import Flask,render_template,request


app = Flask(__name__)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
from PIL import Image
import glob
import pickle
from pickle import dump, load
from time import time
from keras_preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers import add
from keras.applications.inception_v3 import InceptionV3
from keras_preprocessing import image
from keras.models import Model, load_model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences



def preprocess(image_path):
    # Converting all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Converting PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Adding one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocessing the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

def encode(image):
    image = preprocess(image) # preprocessing the image
    fea_vec = model_new.predict(image) # Geting the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshaping from (1, 2048) to (2048, )
    return fea_vec


max_length = 34
vocab_size = 2531
embedding_dim = 200


mdl = load_model('model_weights/weight9.h5')



wordtoix = pickle.load(open('static/wordtoix.pkl','rb'))
ixtoword = pickle.load(open('static/ixtoword.pkl','rb'))



def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = mdl.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/after", methods = ['GET','POST'])
def after():
    file = request.files['file1']
    file.save('static/file.jpg')
    fv = encode('static/file.jpg')
    fv = fv.reshape((1,2048))
    fc = greedySearch(fv)
    return render_template('predict.html',fc = fc)


if __name__ == "_main_":
    app.run(host = '0.0.0.0',port = 5000)