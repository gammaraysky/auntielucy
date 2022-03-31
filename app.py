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


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


global model 
download("http://foxandcat.net/ibm/model_v4.h5", dest_folder="model")

model = tf.keras.models.load_model('model/model_v4.h5')


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
