from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        gender = int(request.form['gender'])

        prediction = model.predict([[age, gender]])
        result = prediction[0]

        return render_template('index.html', prediction_text=f"Suggested Movie Genre: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Invalid input.")

if __name__ == '__main__':
    app.run(debug=True)
