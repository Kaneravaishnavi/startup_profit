from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open("profit_prediction_model.pkl", "rb"))


# Load the column names
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rd = float(request.form['rd'])
        admin = float(request.form['admin'])
        marketing = float(request.form['marketing'])
        state = request.form['state']

        x = np.zeros(len(data_columns))
        x[0] = rd
        x[1] = admin
        x[2] = marketing

        # Handle one-hot encoding manually
        if state == 'Florida':
            if 'state_florida' in data_columns:
                x[data_columns.index('state_florida')] = 1
        elif state == 'New York':
            if 'state_new york' in data_columns:
                x[data_columns.index('state_new york')] = 1
        # No need to set anything if state is California (it's the base case)

        prediction = model.predict([x])[0]

        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"Something went wrong: {e}"


if __name__ == "__main__":
    app.run(debug=True)
