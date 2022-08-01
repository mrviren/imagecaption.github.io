from flask import Flask,render_template,request
import logging

logging.basicConfig(filename='logs.txt',filemode='w',format='%(asctime)s %(message)s',)
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

import pickle
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
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

logger.debug("all dependencies imported")

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


logger.debug("Feature extraction model imported successfully")

max_length = 34
vocab_size = 2531
embedding_dim = 200
# inputs1 = Input(shape=(2048,)) # feature vector
# fe1 = Dropout(0.5)(inputs1)
# fe2 = Dense(256, activation='relu')(fe1)

# inputs2 = Input(shape=(max_length,)) # word sequence
# se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
# se2 = Dropout(0.5)(se1)
# se3 = LSTM(256)(se2)

# decoder1 = add([fe2, se3])
# decoder2 = Dense(256, activation='relu')(decoder1)
# outputs = Dense(vocab_size, activation='softmax')(decoder2)
# mdl = Model(inputs=[inputs1, inputs2], outputs=outputs)

# embedding_matrix = pickle.load(open('static/embedding_matrix.pkl','rb'))
# mdl.layers[2].set_weights([embedding_matrix])
# mdl.layers[2].trainable = False
# mdl.compile(loss='categorical_crossentropy', optimizer = 'adam')

mdl = load_model('model_weights/weight9.h5')



wordtoix = pickle.load(open('static/wordtoix.pkl','rb'))
ixtoword = pickle.load(open('static/ixtoword.pkl','rb'))

logger.debug("Caption generator model imported successfully")

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
    logger.debug("Task completed successfully")
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

logger.debug("Task completed successfully")

if __name__ == "_main_":
    app.run(debug=True)