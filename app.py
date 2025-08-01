from flask import Flask, render_template, request, redirect
from numpy import vectorize
from prediction_pipeline import preprocessing, get_prediction, vectorize_text
from my_logger import logging


app = Flask(__name__)

logging.info('Flask server started')

data = dict()
reviews = []
positive = 0
negative = 0

@app.route('/')
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info('============home page ============')
    return render_template('index.html', data= data)
    

@app.route("/" , methods = ['post'])
def my_post():
    text= request.form['text']
    logging.info(f'Text : {text}')

    preprocessed_txt = preprocessing(text)
    logging.info(f'Preprocessed Text : {preprocessed_txt}')

    vectorized_txt = vectorize_text(preprocessed_txt )
    logging.info(f'vectorized Text : {vectorized_txt}')

    prediction = get_prediction(vectorized_txt)
    logging.info(f'Prediction : {prediction}')

    if prediction == 'negtive':
        global negative
        negative += 1
    else:
        global positive
        positive += 1

    reviews.insert(0, text)
    return redirect(request.url)

if __name__== "__main__":
    app.run()
