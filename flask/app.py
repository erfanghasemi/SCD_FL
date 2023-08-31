from flask import Flask, render_template, request
import os
import sys

sys.path.append('..')
import utils
import torch

app = Flask(__name__)
CHECKPOINTS_PATH = "..\checkpoints"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


UPLOAD_FOLDER = r'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = utils.load_checkpoint(CHECKPOINTS_PATH, str(DEVICE)) 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        image = request.files['image']

        if image.filename == '':
            return redirect(request.url)
        if image:
            # filename = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            filename = app.config['UPLOAD_FOLDER'] + "/" + image.filename
            image.save(filename)
            print(filename)
            results = utils.inference(model, filename)
            print(results)
            # results = "nevus"
            return render_template('index.html', image_filename=image.filename, results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)