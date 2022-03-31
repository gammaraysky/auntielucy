import numpy as np
from flask import Flask, render_template, request, redirect, flash, url_for
import urllib.request
from werkzeug.utils import secure_filename
import os
import predict_veg_fruit as pvg
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
import requests

app = Flask(__name__)


UPLOAD_FOLDER = './upload/'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = tf.keras.models.load_model('model/model_v6.h5')


@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html', thumbnail=url_for('static', filename='images/Apple.jpg'))


@app.route('/predict', methods=['POST'])
def submit_file():
    if request.method == 'POST' and 'file' in request.files:
        image = request.files['file']
        im = Image.open(image)
        im = im.resize((150,150),Image.NEAREST)

        result, confidence, writeup, thumbnail = pvg.predict_image(im, model)
        percent = confidence*100
        
        
        if (confidence>0.7):
            return render_template('index.html', result=result, confidence=f"{percent:.2f}%", writeup=writeup, thumbnail=url_for('static', filename=thumbnail))
        else:
            return render_template('index.html', result=f"I think it's {result} but I'm not so sure, could you try another picture?", confidence=f"{percent:.2f}%", writeup='', thumbnail=thumbnail)



if __name__ == "__main__":
    app.run()
