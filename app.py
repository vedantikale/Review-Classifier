from flask import Flask,render_template,request
import pickle
import pandas as pd
import nltk
from nltk.corpus import  stopwords
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
#from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
multinomial_model = pickle.load(open('multinomial_model.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html') 

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum:
      y.append(i)
  
  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  
  text = y[:]
  y.clear()
  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

    

@app.route('/predict',methods = ['POST'])
def predict():
    review = request.form.get('review')
    review = transform_text(review)
    vector_input = vectorizer.transform([review])
    vector_input = vector_input.reshape(1,-1)
    result1 = model.predict(vector_input)[0] 
    result2 = multinomial_model.predict(vector_input)[0]
    if result2 == 1:
        return "positive review"
    else:
        return "negative review"

if __name__ == '__main__':
    app.run(debug = True)


    
