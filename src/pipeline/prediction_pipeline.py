import pickle

from src.utils import clean_text


class PredictionPipeline:

    def __init__(self):

        self.model_path = "artifacts/model.pkl"
        self.vectorizer_path = "artifacts/vectorizer.pkl"

    def predict(self,text):

        with open(self.model_path,"rb") as f:
            model = pickle.load(f)

        with open(self.vectorizer_path,"rb") as f:
            vectorizer = pickle.load(f)

        text = clean_text(text)

        vector = vectorizer.transform([text]).toarray()

        prediction = model.predict(vector)

        return prediction[0]