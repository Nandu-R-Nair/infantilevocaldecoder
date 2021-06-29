import numpy as np
from flask import Flask, flash, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pickle
import os
import audioread
import sounddevice as sd
import struct
import wave
import contextlib
import sys
import glob
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile as wav
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pydub import AudioSegment 
import math
from pydub.utils import make_chunks
from collections import Counter

app = Flask(__name__)
app.secret_key = "super secret key"
model = pickle.load(open('myRandomForest.pkl', 'rb'))

UPLOAD_FOLDER = 'C:/Users/91902/infantilevocaldecoder/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
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
    dir3 = 'C:/Users/91902/infantilevocaldecoder/audio'
    for f in os.listdir(dir3):
        os.remove(os.path.join(dir3, f))
        
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('extra.html')
            #return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('extra.html')
            #flash('No selected file')
            #return redirect(request.url)
        if file and allowed_file(file.filename):

            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.rename(UPLOAD_FOLDER +'/' + filename, UPLOAD_FOLDER+ '/' + 'output.wav')
            #return 'file uploaded successfully'
        else:
            return render_template('extra.html')

        for f in os.listdir(dir):
            full_path = os.path.join(dir, f)
        sound = AudioSegment.from_mp3(full_path)
        sound.export("C:/Users/91902/infantilevocaldecoder/audio/output.wav", format="wav")
        raw_audio = dict()
        directory = 'C:/Users/91902/infantilevocaldecoder/audio'
        
        for filename in os.listdir(directory):
            if filename.endswith(".wav"): 
                raw_audio[os.path.join(directory, filename)] = 'output'
            else:
                continue
        for audio_file in raw_audio:
            chop_song(audio_file, raw_audio[audio_file])
        predictions = []
        for i, filename in enumerate(os.listdir('C:/Users/91902/infantilevocaldecoder/output/')):
            #last_number_frames = -1
            if filename.endswith(".wav"):
                #print filename
                audiofile, sr = librosa.load("C:/Users/91902/infantilevocaldecoder/output/"+filename)
                fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=40)
                x = pd.DataFrame(fingerprint, dtype = 'float32')
                prediction = model.predict(fingerprint)
                #print prediction
                predictions.append(prediction[0])
                
        data = Counter(predictions)
        data2=data.most_common(1)
        result=data2[0]
        result2=result[0]
        if (result2=='hungry'):
            #dataToRender= 'is Hungry'
            return render_template('result.html',dataToRender='is Hungry')
        if (result2=='pain'):
            #dataToRender= 'is Hungry'
            return render_template('result.html',dataToRender='is in pain')
        else:
            #dataToRender= 'is Hungry'
            return render_template('result.html',dataToRender='is in discomfort')

        
        
        #return result2
def chop_song(filename, folder):
    #print(filename)
    head, tail = os.path.split(filename)
    shortfilename = tail
    #print(shortfilename)
    myaudio = AudioSegment.from_file(filename, "wav") 
    chunk_length_ms = 1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    #Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
        chunk_len=len(chunk)
        if(chunk_len==1000):
            chunk_name = shortfilename + 'snippet' + str(i+1) + '.wav'
            #print ("exporting", chunk_name)
            chunk.export( 'C:/Users/91902/infantilevocaldecoder/'+ folder + '/'+ chunk_name, format="wav")


if __name__ == '__main__':
   app.run(debug = True)