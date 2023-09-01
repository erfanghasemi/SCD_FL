from flask import Flask, render_template, request, redirect, flash, url_for
import os
import sys
sys.path.append('..')
import utils
import torch
from shutil import move


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
            allowed_extensions = {'jpg', 'jpeg'}
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                file_path = app.config['UPLOAD_FOLDER'] + "/" + file.filename 
                file.save(file_path)
                model = utils.load_checkpoint(CHECKPOINTS_PATH, str(DEVICE)) 
                prediction_disease = utils.inference(model, file_path)
                flash(prediction_disease)
                flash(0.1)
                flash(file_path)
                flash(file.filename)
            else:
                flash('Invalid file format. Allowed formats: jpg, jpeg, png')
            return redirect('/')


@app.route('/checkpoint/<filename>', methods=['POST'])
def submit_checkpoint(filename):
    if request.method == 'POST':
        choice = request.form['choice']
        upload_path = r"..\flask\static\uploads"
        training_path = r"..\lesions_dataset\FL_Training_Dataset"
        file_path = os.path.join(upload_path, filename)
        dst_path = os.path.join(training_path, choice)

        if choice == "unknown":
            os.remove(file_path) # comment this line is you need unkonwn images in the future
            redirect('/')

        elif os.path.isfile(file_path):
            try:
                move(file_path, dst_path)
                print(f"{filename} moved to {choice} folder in Training")
            except Exception as e:
                print(f"Error moving {filename}: {str(e)}")
                os.remove(file_path)
        else:
            print(f"File not found: {filename}")
        
        return redirect('/')


if __name__ == "__main__":
    app.run()