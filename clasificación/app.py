import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask,render_template,request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
    

@app.route('/prediccion',methods=['GET','POST'])   
def predict():
    if request.method =='POST':
        print(request.form.get('var_1'))
        print(request.form.get('var_2'))
        print(request.form.get('var_3'))       
        print(request.form.get('var_4'))   
        print(request.form.get('var_5'))   
    try:
        var_1=float(request.form['var_1'])
        var_2=float(request.form['var_2'])
        var_3=float(request.form['var_3'])
        var_4=float(request.form['var_4'])
        var_5=float(request.form['var_5'])
        
        pred_args=[var_1,var_2,var_3,var_4,var_5]
        pred_arr=np.array(pred_args)
        preds=pred_arr.reshape(1,-1)
        model=open("modelo.h5","rb")
        lr_model=load_model("modelo.h5", compile=False)
        model_prediction=lr_model.predict(preds)
        model_prediction=round(float(model_prediction),2)
        if (model_prediction > 0.5):
            res = "Aprueba"
        else:
            res = "No aprueba"
        
    except ValueError:
        return "Por favor entra nombre válidos"
    return render_template('prediccion.html',prediccion=res)
    
if __name__=='__main__':
     app.run(debug=False)