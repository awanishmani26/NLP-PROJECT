from gensim.models import Word2Vec
import pickle
import os


def train_word2vec(sentences):

    tokenized = [sentence.split() for sentence in sentences]

    model = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    os.makedirs("artifacts", exist_ok=True)

    pickle.dump(model, open("artifacts/word2vec.pkl", "wb"))

    return model