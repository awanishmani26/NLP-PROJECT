import os
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from src.components.word2vec_embedding import train_word2vec
from src.utils import clean_text


class DataTransformation:

    def __init__(self):

        self.vectorizer_path = "artifacts/vectorizer.pkl"

    def initiate_data_transformation(self,train_path,test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df['reviewText'] = train_df['reviewText'].apply(clean_text)
        test_df['reviewText'] = test_df['reviewText'].apply(clean_text)

        # Train Word2Vec
        sentences = train_df['reviewText'].tolist()
        w2v_model = train_word2vec(sentences)

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/word2vec.pkl","wb") as f:
            pickle.dump(w2v_model,f)

        X_train = train_df['reviewText']
        y_train = train_df['sentiment']

        X_test = test_df['reviewText']
        y_test = test_df['sentiment']

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2)
        )

        X_train = vectorizer.fit_transform(X_train).toarray()
        X_test = vectorizer.transform(X_test).toarray()

        with open(self.vectorizer_path,"wb") as f:
            pickle.dump(vectorizer,f)

        return X_train,X_test,y_train.values,y_test.values