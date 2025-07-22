import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()

#load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

#load vectorizer
with open('static/model/vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

#load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

#load Vocabulary
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns = ["tweet"])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r"https?:\/\/.*[\r\n]*", "", x, flags=re.MULTILINE).split()))
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"] = data["tweet"].str.replace('\d+','',regex = True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split()if x not in sw))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorize_text(preprocessed_txt):
    return vectorizer.transform(preprocessed_txt)

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1 :
        return 'Negative'
    else: 
        return 'positive'

