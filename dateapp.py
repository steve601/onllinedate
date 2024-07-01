from flask import Flask,render_template,request
import numpy as np
from source.main_project.pipeline.predict_pipeline import UserData,PredicPipeline

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('online.html')

@app.route('/predict',methods=['POST'])
def do_prediction():
    data = UserData(
        gender=request.form.get('gender'),
        purchasedvip=request.form.get('vip'),
        income=request.form.get('income'),
        children=request.form.get('children'),
        age=request.form.get('age'),
        attractiveness=request.form.get('rating')
    )
    user_dataframe = data.get_data_as_df()
    
    predict_pipe = PredicPipeline()
    y_hat = int(np.round(predict_pipe.predict(user_dataframe),0))
    if y_hat == 0:
        msg = 'None matches your profile'
    elif y_hat == 1:
        msg = f"{y_hat} person matches your profile"
    else:
        msg = f"{y_hat} persons matches your profile"
    
    return render_template('online.html',text=msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
    