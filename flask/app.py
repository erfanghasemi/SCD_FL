from flask import Flask, render_template, request, redirect, flash, url_for
import urllib.request
import os

import sys
sys.path.append('..')
import utils
import torch
from shutil import copy, move


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
            file_path = app.config['UPLOAD_FOLDER'] + "/" + file.filename
            file.save(file_path)
            model = utils.load_checkpoint(CHECKPOINTS_PATH, str(DEVICE)) 
            prediction_disease = utils.inference(model, file_path)
            flash(prediction_disease)
            flash(0.1)
            flash(file_path)
            flash(file.filename)
            return redirect('/')


@app.route('/checkpoint/<filename>', methods=['POST'])
def submit_checkpoint(filename):
    if request.method == 'POST':
        choice = request.form['choice']
        
        upload_path = r"..\flask\static\uploads"
        training_path = r"..\lesions_dataset\FL_Training_Dataset"
        
        # Handle the checkpoint submission based on the 'choice' value
        if choice == "unknown":
            redirect('/')
        file_path = os.path.join(upload_path, filename)
        dst_path = os.path.join(training_path, choice)
        move(file_path, dst_path)
        print("{} moved tp {} folder in Traning".format(filename, choice))
        
        return redirect('/')


if __name__ == "__main__":
    app.run()