from flask import Flask, render_template, request, redirect, flash, url_for
import urllib.request
import os

import sys
sys.path.append('..')
import utils
import torch


app = Flask(__name__)
app.secret_key = "super secret key"

CHECKPOINTS_PATH = "..\checkpoints"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UPLOAD_FOLDER = r'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = app.config['UPLOAD_FOLDER'] + "/" + file.filename
            file.save(filename)
            model = utils.load_checkpoint(CHECKPOINTS_PATH, str(DEVICE)) 
            prediction_disease = utils.inference(model, filename)
            flash(prediction_disease)
            flash(0.1)
            flash(filename)
            return redirect('/')


if __name__ == "__main__":
    app.run()