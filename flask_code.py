#from crypt import methods
from fileinput import filename
from importlib.metadata import files
import numpy as np
from flask import Flask, request, jsonify, url_for, render_template,flash,redirect
import pickle
from fastbook import *
from fastai import *
import os
from fastai.vision.widgets import *

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


learn_inf = load_learner('C:/Users/shubh/Desktop/Fun/chess_piece_classifier/model1.pkl')

# #btn_upload = SimpleNamespace(data = ['C:/Users/shubh/Desktop/Fun/chess_piece_classifier/White_knight.jpg'])
# img = PILImage.create(btn_upload.data[-1])
# pred,pred_idx,probs = learn_inf.predict(img)
# lbl_pred = widgets.Label()
# lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
# print(lbl_pred.value)

app = Flask(__name__)

# @app.route('/apitest')
# def apitest():
#     return 'API working'


# @app.route('/sentiment', methods=['POST'])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
# def sentiment():
#    if request.method == 'POST':
#     text_data = pd.DataFrame(request.json)
#     text_out = get_sentiment_DL(prd_model, text_data, word_idx)
 
#     text_out = text_out[['ref','Sentiment_Score']]
#    #Convert df to dict and then to Json
#     text_out_dict = text_out.to_dict(orient='records')
#     text_out_json = json.dumps(text_out_dict, ensure_ascii=False)
#    return text_out_json

# def get_sentiment_DL(best_model, text_data, word_idx):
#     '''Model Processing'''
#     return sentiment_score

UPLOAD_FOLDER = 'C:/Users/shubh/Desktop/Fun/chess_piece_classifier/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods = ['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        #btn_upload = SimpleNamespace(data = file)
        img = PILImage.create(file)
        pred,pred_idx,probs = learn_inf.predict(img)
        lbl_pred = widgets.Label()
        lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
        print(lbl_pred.value)
        
        flash('The image is a ' + lbl_pred.value)
        return render_template('index.html',filename=file.filename)
    else:
        flash('Allowed image types are - png, jpg, and jpeg')
        return redirect(request.url)
    
    
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',filename='chess_piece_classifier/' + filename), code=301)

if __name__ == '__main__':
    app.run()