from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['review']

    pipeline = PredictionPipeline()

    result = pipeline.predict(text)

    if result == 1:
        prediction = "Positive Review 😊"
    else:
        prediction = "Negative Review 😞"

    return render_template("index.html", prediction_text=prediction)


if __name__ == "__main__":
    app.run(host = "0.0.0.0")