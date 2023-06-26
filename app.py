import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash
import numpy as np
import pandas as pd


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    data=np.array(list(data.values()))
    print(data)
    new_data=model.predict(data.reshape(1,-1))
    name=['Setosa','Versicolor','Virginica']
    return jsonify(name[new_data[0]])

if __name__ =="__main__":
    app.run(debug=True)

