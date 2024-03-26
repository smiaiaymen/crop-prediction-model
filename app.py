from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
def predict():
    T = request.form.get('T')
    H = request.form.get('H')
    PH = request.form.get('PH')
    N = request.form.get('N')
    P = request.form.get('P')
    K = request.form.get('K')

    input_query = np.array([[T,H,PH,N,P,K]])
    result = model.predict(input_query)[0]

    return jsonify({'culture': str(result)})

if __name__ == '__main__':
    app.run(debug=True)