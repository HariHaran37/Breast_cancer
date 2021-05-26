#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:51:47 2021

@author: hari
"""

from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle as pkl
sc=pkl.load(open('model/scalar.pkl','rb'))
app=Flask(__name__)
model=pkl.load(open('model/logistic_model.pkl','rb'))


from time import time
start=time()
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def title():
    return render_template('index.html')


@app.route('/')
def main():
    return render_template('prediction.html')

@app.route('/prediction',methods=['POST','GET'])
def home():
    if request.method=='POST':
        a=float(request.form['radius_mean'])
        b=float(request.form['texture_mean'])
        c=float(request.form['smoothness_mean'])
        d=float(request.form['compactness_mean'])
        e=float(request.form['symmetry_mean'])
        f=float(request.form['fractal_dimension_mean'])
        x=[[a,b,c,d,e,f]]
       # print(a,b,c,d)
        data=model.predict(sc.transform(x))
        if data==0:
            data="Benign"
        else:
            data="Malignant"
        
      
        return render_template('result.html',data=data,time='{:2.6}'.format(time()-start))
    else:
        return render_template('prediction.html')




if __name__=='__main__':
    app.run(debug=True)
    


