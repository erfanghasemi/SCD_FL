from flask import Flask, render_template, request, redirect, url_for
import os
import sys

sys.path.append('../')
import utils
import torch

app = Flask(__name__)
CHECKPOINTS_PATH = "checkpoints"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    image = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if image.filename == '':
        return redirect(request.url)

    # Save the uploaded image to the specified directory
    if image:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(filename)
        model = utils.load_checkpoint(CHECKPOINTS_PATH, str(DEVICE)) 
        utils.inference(model, image_path)
        return 'Image uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
