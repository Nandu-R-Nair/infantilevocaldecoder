import numpy as np
from flask import Flask, flash, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pickle
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/91902/infantilevocaldecoder/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
ALLOWED_EXTENSIONS = {'wav'}
@app.route('/')
def home():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    
    dir = 'C:/Users/91902/infantilevocaldecoder/tmp'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
        
    dir2 = 'C:/Users/91902/infantilevocaldecoder/output'
    for f in os.listdir(dir2):
        os.remove(os.path.join(dir2, f))
        
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'file uploaded successfully'

if __name__ == '__main__':
   app.run(debug = True)