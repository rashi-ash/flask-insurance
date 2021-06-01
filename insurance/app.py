import numpy as np
from flask import Flask, request, jsonify, render_template## render_template redirect to the home page in index.html
import pickle


app = Flask(__name__) ## to initialize the flask

with open('model_pickle','rb')as f:
    mp=pickle.load(f)


# define from where the user inout is getting
@app.route('/', methods=['GET','POST'])
def index():
   return render_template("index.html")
# the user input is fed to the model.py to get the predicted value and return the result
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    prediction =mp.predict(final_features)

    #output = round(prediction[0], 2)

    # display the result in same html page
    return render_template("index.html", prediction_text='Prediction that a persion will buy life insurance at this age is :  {}'.format(prediction))



if __name__ == "__main__":
    app.run(debug=True)